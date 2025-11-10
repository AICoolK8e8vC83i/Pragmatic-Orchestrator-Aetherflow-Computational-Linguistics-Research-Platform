"""
Agent_VisionAnalyzer: Multimodal image/video analysis
Model: Gemini 2.5 Flash via Google AI API + Google Cloud Vision API
Theory: Visual semantics + multimodal pragmatics
"""

import google.generativeai as genai
from google.cloud import vision
import base64
import os
import io
import ollama
from typing import Optional, Dict, Tuple, Any
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

class Agent_VisionAnalyzer:
    """
    Analyzes images and videos using vision-language model.
    Model: Gemini 2.5 Flash
    """
    
    def __init__(self):
        self.model_name = "gemini-2.5-flash"
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
        self.model = None
        self.vision_client = None  # Google Cloud Vision API client
        self.parser_model = "gemma3:1b"  # Gemma 3 1B for parsing/formatting truncated responses
        self._initialize_api()
    
    def _initialize_api(self):
        """Initialize both Gemini and Cloud Vision API clients."""
        # Initialize Gemini
        gemini_ok = False
        if not self.api_key:
            print(f"⚠️  Gemini API key (GEMINI_API_KEY) not found in environment")
            print(f"   Please set GEMINI_API_KEY in .env file")
        else:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print(f"✅ Gemini 2.5 Flash ({self.model_name}) is available")
                gemini_ok = True
            except Exception as e:
                print(f"⚠️  Gemini API initialization failed: {e}")
                print(f"   Make sure GEMINI_API_KEY is set correctly")
        
        # Initialize Cloud Vision API
        vision_ok = False
        try:
            # Check for credentials - can be from env var or default location
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if creds_path and os.path.exists(creds_path):
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                print(f"✅ Using Cloud Vision credentials: {creds_path}")
            else:
                # Try to find JSON file in project root (common locations)
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
                # Look for JSON files with common patterns
                json_files = []
                if os.path.exists(project_root):
                    for f in os.listdir(project_root):
                        if f.endswith('.json'):
                            # Prioritize files with 'google', 'vision', 'gcp', 'cloud' in name
                            if any(keyword in f.lower() for keyword in ['google', 'vision', 'gcp', 'cloud', 'service', 'account']):
                                json_files.append(f)
                            # Also check for service account key pattern (long alphanumeric names)
                            elif len(f.replace('.json', '')) > 20 and any(c.isdigit() for c in f):
                                json_files.append(f)
                
                if json_files:
                    # Use first matching file
                    creds_path = os.path.join(project_root, json_files[0])
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
                    print(f"✅ Found Cloud Vision credentials: {creds_path}")
                else:
                    print(f"⚠️  No Google Cloud credentials JSON found in project root")
                    print(f"   Set GOOGLE_APPLICATION_CREDENTIALS env var or place JSON file in: {project_root}")
            
            # Initialize client (will use GOOGLE_APPLICATION_CREDENTIALS)
            self.vision_client = vision.ImageAnnotatorClient()
            print(f"✅ Google Cloud Vision API is available")
            vision_ok = True
        except Exception as e:
            print(f"⚠️  Cloud Vision API initialization failed: {e}")
            print(f"   Make sure GOOGLE_APPLICATION_CREDENTIALS is set or JSON file is in project root")
            print(f"   Error details: {str(e)}")
            self.vision_client = None
        
        return gemini_ok or vision_ok  # Return True if at least one works
    
    def _base64_to_image(self, image_base64: str) -> Optional[Image.Image]:
        """Convert base64 string to PIL Image."""
        try:
            image_data = base64.b64decode(image_base64)
            img = Image.open(io.BytesIO(image_data))
            print(f"✅ [Gemini] Base64 image decoded successfully")
            return img
        except Exception as e:
            print(f"❌ [Gemini] Base64 to image conversion failed: {e}")
            return None
    
    def _parse_with_gemma(self, truncated_content: str, original_query: str) -> str:
        """Use Gemma 3 1B to parse, complete, and format truncated Gemini response."""
        try:
            print(f"🔄 [Gemma 3 1B] Parsing and formatting truncated response...")
            
            parse_prompt = f"""The following is a truncated image analysis from Gemini 2.5 Flash. 
The response was cut off mid-sentence (finish_reason: MAX_TOKENS).

Original query: "{original_query}"

Truncated Gemini response:
{truncated_content}

Your task:
1. Keep ALL of Gemini's analysis - do not delete anything
2. If the response is incomplete, add a brief conclusion (1-2 sentences)
3. Format the entire response in clear markdown with proper headings, bullet points, and structure
4. Make it readable and well-organized
5. Preserve all technical details and observations from Gemini

Output the complete, formatted analysis in markdown:"""
            
            result = ollama.generate(
                model=self.parser_model,
                prompt=parse_prompt,
                options={"temperature": 0.3, "num_predict": 500}
            )
            
            parsed = result.get("response", "").strip()
            if parsed:
                print(f"✅ [Gemma 3 1B] Successfully parsed and formatted response")
                return parsed
            else:
                print(f"⚠️  [Gemma 3 1B] Empty response, using original truncated content")
                return truncated_content
                
        except Exception as e:
            print(f"⚠️  [Gemma 3 1B] Parsing failed: {e}")
            print(f"   Using original truncated content")
            return truncated_content
    
    def _extract_content_from_response(self, response) -> Tuple[Optional[str], bool]:
        """
        Extract content from Gemini response, handling truncation.
        Returns: (content, is_truncated)
        """
        content = None
        is_truncated = False
        
        # Check finish_reason first to detect truncation
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            
            # Check if truncated (finish_reason: 2 = MAX_TOKENS)
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                if finish_reason == 2:  # MAX_TOKENS
                    is_truncated = True
                    print(f"⚠️  [Gemini] Response truncated (finish_reason: MAX_TOKENS)")
        
        # First try response.text
        try:
            content = response.text
            print(f"✅ [Gemini] Extracted content via response.text")
        except Exception as text_error:
            print(f"⚠️  [Gemini] Could not access response.text: {text_error}")
            
            # Try to extract from candidate.content.parts
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        # Try different ways to get text
                        if hasattr(part, 'text') and part.text:
                            text_parts.append(part.text)
                        elif hasattr(part, 'inline_data') and hasattr(part.inline_data, 'data'):
                            # Skip binary data
                            continue
                        elif str(part):  # Fallback to string representation
                            part_str = str(part)
                            if part_str and not part_str.startswith('<'):
                                text_parts.append(part_str)
                    
                    if text_parts:
                        content = " ".join(text_parts)
                        print(f"✅ [Gemini] Extracted content from candidate.parts ({len(text_parts)} parts)")
                    else:
                        print(f"⚠️  [Gemini] No text parts found in candidate.content.parts")
                        # Try to get raw string representation
                        try:
                            content = str(candidate.content)
                            if content and len(content) > 10:
                                print(f"✅ [Gemini] Extracted content from string representation")
                        except:
                            pass
        
        return content, is_truncated
    
    def _analyze_with_cloud_vision(self, image_path: str) -> Dict[str, Any]:
        """
        Use Google Cloud Vision API for OCR and label detection.
        Returns structured data with text, labels, and safe search.
        """
        if not self.vision_client:
            return {"text": "", "labels": [], "error": "Vision client not initialized"}
        
        try:
            # Load image bytes
            with io.open(image_path, "rb") as image_file:
                content = image_file.read()
            
            image = vision.Image(content=content)
            
            # 1. Text Detection (OCR)
            print(f"🔍 [Cloud Vision] Running OCR...")
            text_response = self.vision_client.text_detection(image=image)
            detected_text = ""
            if text_response.text_annotations:
                detected_text = text_response.text_annotations[0].description.strip()
                print(f"✅ [Cloud Vision] OCR detected {len(detected_text)} characters")
            
            # 2. Label Detection
            print(f"🔍 [Cloud Vision] Detecting labels...")
            label_response = self.vision_client.label_detection(image=image)
            labels = []
            if label_response.label_annotations:
                labels = [(label.description, label.score) for label in label_response.label_annotations]
                print(f"✅ [Cloud Vision] Found {len(labels)} labels")
            
            # 3. Safe Search (optional, for content filtering)
            safe_search = None
            try:
                safe_response = self.vision_client.safe_search_detection(image=image)
                safe_search = safe_response.safe_search_annotation
            except:
                pass
            
            return {
                "text": detected_text,
                "labels": labels,
                "safe_search": safe_search,
                "error": None
            }
            
        except Exception as e:
            print(f"⚠️  [Cloud Vision] Analysis failed: {e}")
            return {"text": "", "labels": [], "error": str(e)}
    
    def process(self, query: str, context: list = None, image_path: Optional[str] = None, image_base64: Optional[str] = None):
        """
        Process vision query with image.
        
        Args:
            query: Text query about the image
            context: Previous conversation context
            image_path: Path to image file
            image_base64: Base64 encoded image (alternative to image_path)
        """
        print(f"🏷️  Agent Badge: Vision Analyzer (Multimodal Pragmatics)")
        if not self.model:
            return {"error": "Gemini API not initialized", "analysis": "", "ai_success": False}
        
        # Get image as PIL Image object
        img = None
        
        if image_base64:
            # Convert base64 to PIL Image
            img = self._base64_to_image(image_base64)
        elif image_path:
            # Load from file path
            if not os.path.exists(image_path):
                # Try to extract image path from query
                import re
                path_match = re.search(r'(["\']?)([^\s"\']+\.(jpg|jpeg|png|gif|webp))\1', query, re.IGNORECASE)
                if path_match:
                    image_path = path_match.group(2)
            
            if image_path and os.path.exists(image_path):
                try:
                    img = Image.open(image_path)
                    print(f"✅ [Gemini] Image loaded: {img.size[0]}x{img.size[1]} pixels, mode: {img.mode}")
                except Exception as e:
                    print(f"❌ [Gemini] Failed to load image: {e}")
                    return {"error": f"Failed to load image: {e}", "analysis": "", "ai_success": False}
        
        if not img:
            return {"error": "No valid image provided", "analysis": "", "ai_success": False}
        
        # Step 1: Use Cloud Vision API for OCR and labels (if available)
        vision_data = {}
        actual_image_path = None
        
        if image_path and os.path.exists(image_path):
            actual_image_path = image_path
        elif image_base64:
            # Save base64 to temp file for Cloud Vision
            import tempfile
            image_data = base64.b64decode(image_base64)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(image_data)
                actual_image_path = tmp_file.name
        
        if actual_image_path and self.vision_client:
            vision_data = self._analyze_with_cloud_vision(actual_image_path)
            if actual_image_path != image_path:  # Clean up temp file
                try:
                    os.unlink(actual_image_path)
                except:
                    pass
        
        # If Gemini is not available but we have Cloud Vision data, return that
        if not self.model:
            if vision_data.get("text") or vision_data.get("labels"):
                analysis = ""
                if vision_data.get("text"):
                    analysis += f"**OCR Text (Cloud Vision API):**\n{vision_data['text']}\n\n"
                if vision_data.get("labels"):
                    labels_str = "\n".join([f"- {label[0]} (confidence: {label[1]:.2f})" for label in vision_data['labels']])
                    analysis += f"**Detected Labels (Cloud Vision API):**\n{labels_str}\n"
                return {
                    "analysis": analysis,
                    "query": query,
                    "ai_success": True,
                    "model": "Google Cloud Vision API",
                    "vision_data": vision_data
                }
            else:
                return {"error": "Neither Gemini nor Cloud Vision available", "analysis": "", "ai_success": False}
        
        # Step 2: Build enhanced prompt with Cloud Vision data
        vision_context = ""
        if vision_data.get("text"):
            vision_context += f"\n\n**OCR Text Detected (Cloud Vision API):**\n{vision_data['text']}\n"
        if vision_data.get("labels"):
            labels_str = ", ".join([f"{label[0]} ({label[1]:.2f})" for label in vision_data['labels'][:10]])
            vision_context += f"\n**Labels Detected (Cloud Vision API):** {labels_str}\n"
        
        print(f"🔄 [Gemini 2.5 Flash] Analyzing image with Cloud Vision context...")
        print(f"   🤖 Model: Gemini 2.5 Flash (Google AI API)")
        if vision_context:
            print(f"   📋 Cloud Vision: OCR + {len(vision_data.get('labels', []))} labels")
        
        try:
            # Enhanced prompt for detailed visual analysis with Cloud Vision context
            if query:
                detailed_prompt = f"""{query}{vision_context}

Please provide an extremely detailed visual analysis building upon the Cloud Vision OCR and labels above. Include:

1. **Character Analysis**: Use the OCR text above as a starting point. For Chinese characters, provide pinyin, meaning, stroke order, and cultural significance. Analyze how the characters relate to the visual composition.
2. **Color Analysis**: Describe all colors, shades, gradients, and color relationships in detail. Note any symbolic color meanings and how colors interact with the detected labels.
3. **Symbol Analysis**: Identify all symbols, seals, marks, decorative elements, and their cultural/artistic significance. Cross-reference with the detected labels.
4. **Visual Description**: Provide a comprehensive visual description including composition, layout, spatial relationships, textures, brushwork techniques, and artistic style. Explain how the visual elements relate to the detected text and labels.
5. **Fine Details**: Note subtle details like brush strokes, ink variations, paper texture, aging effects, and any minute elements that contribute to the overall meaning.

Go beyond broad summaries - analyze every visible element in detail, using the Cloud Vision data as context."""
            else:
                detailed_prompt = f"""Analyze this image in extreme detail, building upon the Cloud Vision API data below:{vision_context}

Provide:

1. **Character Analysis**: Use the OCR text above as a starting point. For Chinese characters, provide pinyin, meaning, stroke order, and cultural significance. Analyze how the characters relate to the visual composition.
2. **Color Analysis**: Describe all colors, shades, gradients, and color relationships in detail. Note any symbolic color meanings and how colors interact with the detected labels.
3. **Symbol Analysis**: Identify all symbols, seals, marks, decorative elements, and their cultural/artistic significance. Cross-reference with the detected labels.
4. **Visual Description**: Provide a comprehensive visual description including composition, layout, spatial relationships, textures, brushwork techniques, and artistic style. Explain how the visual elements relate to the detected text and labels.
5. **Fine Details**: Note subtle details like brush strokes, ink variations, paper texture, aging effects, and any minute elements that contribute to the overall meaning.

Go beyond broad summaries - analyze every visible element in detail."""
            
            # Generate content - NO max_output_tokens limit to allow full detailed analysis
            response = self.model.generate_content(
                [detailed_prompt, img],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    # Removed max_output_tokens to allow unlimited detailed analysis
                )
            )
            
            # Extract content from response
            content, is_truncated = self._extract_content_from_response(response)
            
            if not content:
                # Check for blocking
                error_msg = "Could not extract content from response"
                if hasattr(response, 'prompt_feedback') and hasattr(response.prompt_feedback, 'block_reason'):
                    if response.prompt_feedback.block_reason != 0:
                        error_msg = f"Response blocked. Reason: {response.prompt_feedback.block_reason}"
                print(f"❌ [Gemini] {error_msg}")
                return {"error": error_msg, "analysis": "", "ai_success": False}
            
            # If truncated, use Gemma 3 1B to parse and format
            if is_truncated or len(content) < 100:
                print(f"🔄 [Gemini] Response appears truncated or incomplete, using Gemma 3 1B to parse/format...")
                content = self._parse_with_gemma(content, detailed_prompt)
            
            print(f"✅ [Gemini 2.5 Flash] AI call successful, response length: {len(content)} chars")
            
            return {
                "analysis": content,
                "query": query,
                "ai_success": True,
                "model": self.model_name,
                "vision_data": vision_data  # Include Cloud Vision OCR and labels
            }
            
        except Exception as e:
            print(f"❌ [Gemini 2.5 Flash] AI call failed: {e}")
            import traceback
            traceback.print_exc()
            
            # If Gemini fails but we have Cloud Vision data, return that
            if vision_data.get("text") or vision_data.get("labels"):
                analysis = f"⚠️ Gemini analysis failed, but Cloud Vision API provided:\n\n"
                if vision_data.get("text"):
                    analysis += f"**OCR Text (Cloud Vision API):**\n{vision_data['text']}\n\n"
                if vision_data.get("labels"):
                    labels_str = "\n".join([f"- {label[0]} (confidence: {label[1]:.2f})" for label in vision_data['labels']])
                    analysis += f"**Detected Labels (Cloud Vision API):**\n{labels_str}\n"
                return {
                    "analysis": analysis,
                    "query": query,
                    "ai_success": True,  # Partial success with Cloud Vision
                    "model": "Google Cloud Vision API (Gemini failed)",
                    "vision_data": vision_data,
                    "error": f"Gemini failed: {str(e)}"
                }
            
            return {"error": str(e), "analysis": "", "ai_success": False}
