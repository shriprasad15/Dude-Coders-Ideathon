import asyncio
import os
import sys

# Ensure backend module can be imported
sys.path.append(os.path.join(os.getcwd()))

from backend.gemini_graph import run_agent

async def main():
    print("--- Testing Gemini Graph Agent ---")
    
    # Test 1: Simple Text Query
    print("\n[Test 1] Text Query: 'Hello, explain who you are.'")
    try:
        response = await run_agent("Hello, explain who you are.")
        print(f"Agent Response: {response}")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Image Query (if image exists)
    # The user provided an image in the request, let's use it if possible.
    # The metadata showed an uploaded image at:
    # /home/sports/.gemini/antigravity/brain/7ae35ce6-e687-4b43-82a9-305348b0d27a/uploaded_image_1766684370662.png
    # I'll copy it to the local dir for testing or just reference it.
    
    image_path = "/home/sports/.gemini/antigravity/brain/7ae35ce6-e687-4b43-82a9-305348b0d27a/uploaded_image_1766684370662.png"
    if os.path.exists(image_path):
        print(f"\n[Test 2] Image Query with {image_path}")
        print("Query: 'What is this image about?'")
        try:
            response = await run_agent("What is this image about?", image_path=image_path)
            print(f"Agent Response: {response}")
        except Exception as e:
             # Full stack trace for debugging
            import traceback
            traceback.print_exc()
            print(f"Error: {e}")
    else:
        print(f"\n[Test 2] Skipped: Image not found at {image_path}")

if __name__ == "__main__":
    asyncio.run(main())
