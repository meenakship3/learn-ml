import asyncio
import os
import sys
from youtube_transcript_api import YouTubeTranscriptApi
from google import genai
from google.genai import types
from datetime import datetime

# Initialize Gemini client
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
except KeyError:
    print("API key not found")
    sys.exit(1)
else:
    client = genai.Client()


# Generate study guide tool
def generate_study_guide(video_transcript: str, video_id: str) -> str:
    print(f"Generating study guide for video {video_id}...")

    model = "gemini-2.5-flash"
    prompt = f"""Here is a video transcript:
{video_transcript}

Generate a comprehensive study guide in markdown format with the following structure:

# Study Guide: [Video Title/Topic]

## Overview
[Brief summary of the video content]

## Key Concepts
[List and explain important concepts covered]

## Technical Terms
[Define all technical terms mentioned]

## Important Points
[Bullet points of crucial information]

## Summary
[Concise recap of main takeaways]

## Additional Resources
[Suggest related topics for further learning]

Ensure that concepts are explained in a way that a 15-year-old can understand. All technical terms covered in the video must be present and clearly defined. You may search the web for up-to-date information on the topic and fill in useful details if necessary."""
    
    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    response = client.models.generate_content(
        model=model, contents=prompt, config=config
    )

    return response.text


# Get video transcript
def get_transcript(video_id: str, languages: list = None) -> str:
    if languages is None:
        languages = ["en"]
    try:
        yt_api = YouTubeTranscriptApi()
        fetched_transcript = yt_api.fetch(video_id, languages=languages)
        transcript_text = " ".join(snippet.text for snippet in fetched_transcript)

        return transcript_text
    except Exception as e:
        from youtube_transcript_api._errors import (
            CouldNotRetrieveTranscript,
            VideoUnavailable,
            InvalidVideoId,
            NoTranscriptFound,
            TranscriptsDisabled,
        )

        if isinstance(e, NoTranscriptFound):
            error_msg = (
                f"No transcript found for video {video_id} in languages: {languages}"
            )
        elif isinstance(e, VideoUnavailable):
            error_msg = f"Video {video_id} is unavailable"
        elif isinstance(e, InvalidVideoId):
            error_msg = f"Invalid video ID: {video_id}"
        elif isinstance(e, TranscriptsDisabled):
            error_msg = f"Transcripts are disabled for video {video_id}"
        elif isinstance(e, CouldNotRetrieveTranscript):
            error_msg = f"Could not retrieve transcript: {str(e)}"
        else:
            error_msg = f"An unexpected error occurred: {str(e)}"

        print(f"Error: {error_msg}")
        raise Exception(error_msg) from e


# Save study guide to markdown file
def save_study_guide(content: str, lecture_number: int) -> str:
    """Save study guide content to a markdown file in the notes directory."""
    # Create notes directory if it doesn't exist
    notes_dir = "notes"
    os.makedirs(notes_dir, exist_ok=True)
    
    # Generate filename with lecture number
    filename = f"Lecture{lecture_number}.md"
    filepath = os.path.join(notes_dir, filename)
    
    # Write content to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Study guide saved to: {filepath}")
    return filepath


# Main function
async def main():
    video_id_list = [
        "w-pPYzZWXNw",
        "hqHHp7J3DLA",
        "BBLWCFlECyM",
        "Dfc3xSHEbrk",
        "XTkPQRqWbYk",
        "fJPIgGLOtI8",
        "cGmqi2O1cls",
    ]
    
    print(f"Processing {len(video_id_list)} videos...\n")
    
    for i, video_id in enumerate(video_id_list, 1):
        try:
            print(f"[{i}/{len(video_id_list)}] Processing video: {video_id} -> Lecture{i}")
            
            # Get transcript
            transcript = get_transcript(video_id)
            print(f"✓ Transcript retrieved ({len(transcript)} characters)")
            
            # Generate study guide
            study_guide = generate_study_guide(transcript, video_id)
            print(f"✓ Study guide generated ({len(study_guide)} characters)")
            
            # Save to markdown file
            filepath = save_study_guide(study_guide, i)
            print(f"✓ Saved to {filepath}\n")
            
        except Exception as e:
            print(f"✗ Error processing video {video_id}: {str(e)}\n")
            continue
    
    print("\n🎉 All study guides have been generated and saved to the notes directory!")


if __name__ == "__main__":
    asyncio.run(main())
