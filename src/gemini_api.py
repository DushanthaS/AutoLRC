import asyncio
import logging
import google.generativeai as genai

async def get_gemini_transcript(audio_path, api_key, config):
    """Gets plain text transcript from Gemini."""
    logging.info("🎤 Transcribing with Gemini ...")
    max_retries = config.get("MAX_RETRIES", 3)
    retry_delay = config.get("RETRY_DELAY", 5)
    language = config.get("LANGUAGE", "English")
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(config.get("GEMINI_MODEL", "gemini-2.0-flash-thinking"))
    
    instructions = f"""
    You are a highly skilled and meticulous transcription specialist, fluent in {language}. Your sole task is to transcribe spoken {language} audio.
    Transcription: Carefully listen to the audio and create a complete and accurate transcription of the lyrics in {language}.
    Output: You will ONLY provide the complete plain text transcript. Do not include any introductory text, timestamps, explanations, or markdown formatting.
    Important Reminders:
    Focus exclusively on transcribing the spoken {language}. Do not add interpretations, translations into other languages, or any information not directly present in the audio.
    Double-check for spelling errors and accurate {language} script.
    Your entire output should ONLY be the plain text of the transcript.
    Begin!
    """
    
    for attempt in range(max_retries):
        try:
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
                
            # Properly await the async function
            response = await model.generate_content_async(
                [
                    instructions,
                    {"mime_type": "audio/wav", "data": audio_data}
                ]
            )
            
            if response and response.text:
                return response.text.strip()
                
        except Exception as e:
            logging.error(f"❌ Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                logging.info(f"⏳ Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
            else:
                logging.error("❌ All retry attempts failed")
                return None
    
    return None 