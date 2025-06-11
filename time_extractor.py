import yt_dlp
import cv2
import pytesseract
import re
import os
import json
from datetime import timedelta
from typing import List, Dict, Optional, Tuple
import numpy as np

class GameTimestampExtractor:
    def __init__(self, output_dir: str = "temp_videos"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # OCR configuration optimized for scoreboards
        self.ocr_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZ '
        
    def download_video(self, youtube_url: str) -> Tuple[str, Dict]:
        """Download YouTube video and return local file path"""
        ydl_opts = {
            'format': 'best[height<=720]',  # Limit quality for faster processing
            'outtmpl': f'{self.output_dir}/%(title)s.%(ext)s',
            'noplaylist': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=True)
                
                # Check if extraction was successful
                if info is None:
                    raise ValueError(f"Failed to extract video info from {youtube_url}")
                
                # Ensure required fields exist
                if 'title' not in info:
                    raise ValueError("Video info missing title field")
                
                video_path = ydl.prepare_filename(info)
                
                # Get actual downloaded filename (extension might differ)
                actual_path = None
                for file in os.listdir(self.output_dir):
                    if info['title'] in file:
                        actual_path = os.path.join(self.output_dir, file)
                        break
                
                if actual_path is None:
                    # Fallback: use the prepared filename
                    actual_path = video_path
                
                # Verify file actually exists
                if not os.path.exists(actual_path):
                    raise FileNotFoundError(f"Downloaded video not found at {actual_path}")
                    
                return actual_path, info
                
        except Exception as e:
            raise Exception(f"Error downloading video from {youtube_url}: {str(e)}")
    
    def extract_frames(self, video_path: str, interval_seconds: int = 2) -> List[Tuple[float, np.ndarray]]:
        """Extract frames from video at specified intervals"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_seconds)
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                frames.append((timestamp, frame))
                
            frame_count += 1
        
        cap.release()
        return frames
    
    def preprocess_frame_for_ocr(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame to improve OCR accuracy for game clock"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Increase contrast for better text detection
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Threshold to get white text on black background
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    def extract_game_time_from_text(self, text: str) -> Optional[str]:
        """Extract game time from OCR text using regex patterns"""
        # Common NBA game clock patterns - looking for times that make sense as game clocks
        patterns = [
            r'(\d{1,2}:\d{2})',  # MM:SS or M:SS
            r'(\d{1,2}:\d{2}\.\d)',  # MM:SS.T or M:SS.T (with tenths)
        ]
        
        found_times = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Validate time format (max 12:00 for NBA quarters)
                try:
                    time_parts = match.split(':')
                    minutes = int(time_parts[0])
                    seconds_part = time_parts[1].split('.')[0]  # Remove tenths if present
                    seconds = int(seconds_part)
                    
                    # Game clock should be 0-12 minutes, 0-59 seconds
                    if 0 <= minutes <= 12 and 0 <= seconds <= 59:
                        total_seconds = minutes * 60 + seconds
                        found_times.append((match, total_seconds, minutes))
                except:
                    continue
        
        if not found_times:
            return None
            
        # If we found multiple times, prioritize the game clock over shot clock
        # Strategy: prefer times that are either >24 seconds OR have >0 minutes
        game_clock_candidates = [
            (time, total, mins) for time, total, mins in found_times 
            if total > 24 or mins > 0
        ]
        
        if game_clock_candidates:
            # Return the longest time (most likely to be game clock)
            return max(game_clock_candidates, key=lambda x: x[1])[0]
        else:
            # If no clear game clock candidate, return the longest time found
            # This handles cases where game time is genuinely under 24 seconds
            return max(found_times, key=lambda x: x[1])[0]
    
    def extract_quarter_info(self, text: str) -> Optional[int]:
        """Extract quarter information from OCR text"""
        # Look for quarter indicators - handles both numeric and ordinal formats
        quarter_patterns = [
            r'(\d)(?:ST|ND|RD|TH)\s*(?:QTR|QUARTER)?',  # 1ST, 2ND, 3RD, 4TH
            r'Q(\d)',  # Q1, Q2, Q3, Q4
            r'QUARTER\s*(\d)',  # QUARTER 1, QUARTER 2, etc.
            r'(\d)(?:st|nd|rd|th)\s*(?:qtr|quarter)?',  # lowercase versions
        ]
        
        text_upper = text.upper()
        for pattern in quarter_patterns:
            matches = re.findall(pattern, text_upper)
            if matches:
                try:
                    quarter = int(matches[0])
                    if 1 <= quarter <= 4:
                        return quarter
                except:
                    continue
        
        # Check for overtime variations
        overtime_patterns = ['OT', 'OVERTIME', '1OT', '2OT', '3OT']
        for ot_pattern in overtime_patterns:
            if ot_pattern in text_upper:
                return 5  # Represent OT as quarter 5
                
        return None
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """Process a single frame to extract game information"""
        preprocessed = self.preprocess_frame_for_ocr(frame)
        
        # Perform OCR
        text = pytesseract.image_to_string(preprocessed, config=self.ocr_config)
        
        # Extract game time and quarter
        game_time = self.extract_game_time_from_text(text)
        quarter = self.extract_quarter_info(text)
        
        return {
            'game_time': game_time,
            'quarter': quarter,
            'raw_text': text.strip(),
            'confidence': self.calculate_confidence(game_time, quarter)
        }
    
    def calculate_confidence(self, game_time: Optional[str], quarter: Optional[int]) -> float:
        """Calculate confidence score for extracted data"""
        confidence = 0.0
        
        if game_time:
            confidence += 0.7
        if quarter:
            confidence += 0.3
            
        return confidence
    
    def smooth_timeline(self, timeline: List[Dict]) -> List[Dict]:
        """Apply smoothing to remove OCR errors and fill gaps"""
        smoothed = []
        
        for i, entry in enumerate(timeline):
            # If current entry has low confidence, try to interpolate
            if entry['confidence'] < 0.5:
                # Look for nearby high-confidence entries
                prev_good = next((timeline[j] for j in range(i-1, -1, -1) 
                                if timeline[j]['confidence'] >= 0.7), None)
                next_good = next((timeline[j] for j in range(i+1, len(timeline)) 
                                if timeline[j]['confidence'] >= 0.7), None)
                
                if prev_good and next_good:
                    # Simple interpolation (could be more sophisticated)
                    entry['game_time'] = prev_good['game_time']
                    entry['quarter'] = prev_good['quarter']
                    entry['interpolated'] = True
            
            smoothed.append(entry)
        
        return smoothed
    
    def process_video(self, youtube_url: str, frame_interval: int = 2) -> Dict:
        """Main processing function"""
        print(f"Processing video: {youtube_url}")
        
        # Download video
        print("Downloading video...")
        video_path, video_info = self.download_video(youtube_url)
        
        # Extract frames
        print("Extracting frames...")
        frames = self.extract_frames(video_path, frame_interval)
        print(f"Extracted {len(frames)} frames")
        
        # Process each frame
        print("Processing frames for game timestamps...")
        timeline = []
        
        for i, (video_timestamp, frame) in enumerate(frames):
            if i % 10 == 0:  # Progress update
                print(f"Processing frame {i+1}/{len(frames)}")
                
            frame_data = self.process_frame(frame)
            frame_data['video_timestamp'] = video_timestamp
            timeline.append(frame_data)
        
        # Apply smoothing
        print("Smoothing timeline...")
        timeline = self.smooth_timeline(timeline)
        
        # Clean up downloaded video
        if os.path.exists(video_path):
            os.remove(video_path)
        
        return {
            'video_info': {
                'title': video_info.get('title', 'Unknown Title'),
                'duration': video_info.get('duration', 0),
                'upload_date': video_info.get('upload_date', 'Unknown'),
                'video_id': video_info.get('id', 'Unknown')
            },
            'timeline': timeline,
            'processing_stats': {
                'total_frames': len(frames),
                'high_confidence_frames': len([t for t in timeline if t['confidence'] >= 0.7]),
                'frame_interval': frame_interval
            }
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save processing results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")

# Example usage and testing
def main():
    # Example NBA highlight URL (replace with actual URL)
    youtube_url = "https://www.youtube.com/watch?v=8BE1zcoQmKU"
    
    processor = GameTimestampExtractor()
    
    try:
        results = processor.process_video(youtube_url, frame_interval=3)
        processor.save_results(results, "game_timeline.json")
        
        # Print summary
        print("\n=== Processing Summary ===")
        print(f"Video: {results['video_info']['title']}")
        print(f"Duration: {results['video_info']['duration']} seconds")
        print(f"Frames processed: {results['processing_stats']['total_frames']}")
        print(f"High confidence detections: {results['processing_stats']['high_confidence_frames']}")
        
        # Show sample detections
        print("\n=== Sample Timeline ===")
        for entry in results['timeline'][:10]:
            conf_str = f"(confidence: {entry['confidence']:.2f})"
            interp_str = " [INTERPOLATED]" if entry.get('interpolated') else ""
            print(f"Video {entry['video_timestamp']:.1f}s -> Q{entry['quarter']} {entry['game_time']} {conf_str}{interp_str}")
            
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main()
