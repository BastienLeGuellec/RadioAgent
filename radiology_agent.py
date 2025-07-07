import openai
import os
import io
import base64
import glob
from PIL import Image
import numpy as np
import math
from dotenv import load_dotenv
import sys
import google.generativeai as genai

class RadiologyAgent:
    def __init__(self, case_data_path, model_type="openai"):
        self.model_type = model_type
        if self.model_type == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_type == "gemini":
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_model_text = genai.GenerativeModel("gemini-2.5-pro")
            self.gemini_model_vision = genai.GenerativeModel("gemini-2.5-pro")
        else:
            raise ValueError("Invalid model_type. Must be 'openai' or 'gemini'.")

        self.case_data_path = case_data_path
        self.image_paths = self._load_image_paths()
        self.current_slice_index = 0 # 0-indexed
        self.max_slices = len(self.image_paths)
        self.messages = [
            {"role": "system", "content": "You are a helpful radiology assistant. You can respond with 'UP' or 'DOWN' as the last word of your message to request a slice change, or 'DIAGNOSIS:' followed by your diagnosis."}
        ]

    def _load_image_paths(self):
        image_pattern = os.path.join(self.case_data_path, "*.jpg")
        paths = glob.glob(image_pattern)
        paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        return paths

    def _encode_image(self, image_path):
        try:
            with open(image_path, "rb") as image_file:
                os.system(f"viu {image_path} --height 50")
                image_data = image_file.read()
                encoded_string = base64.b64encode(image_data).decode('utf-8')
            return encoded_string
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None

    def _get_current_image_content(self):
        if not self.image_paths:
            return []
        
        current_image_path = self.image_paths[self.current_slice_index]
        encoded_image = self._encode_image(current_image_path)
        
        if encoded_image:
            return [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        return []

    def move_slice(self, direction):
        if direction.upper() == "DOWN":
            if self.current_slice_index < self.max_slices - 1:
                self.current_slice_index += 1
                return f"Moved to slice {self.current_slice_index + 1}/{self.max_slices}. Current image updated."
            else:
                return f"Already at the last slice ({self.current_slice_index + 1}/{self.max_slices}). Cannot move further UP."
        elif direction.upper() == "UP":
            if self.current_slice_index > 0:
                self.current_slice_index -= 1
                return f"Moved to slice {self.current_slice_index + 1}/{self.max_slices}. Current image updated."
            else:
                return f"Already at the first slice ({self.current_slice_index + 1}/{self.max_slices}). Cannot move further DOWN."
        else:
            return "Invalid direction. Use 'UP' or 'DOWN'."

    def get_llm_response(self):
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model="o3", 
                    messages=self.messages,
                )
                agent_response = response.choices[0].message.content.strip()
            elif self.model_type == "gemini":
                # Gemini expects content in a specific format for multimodal input
                gemini_content = []
                for message in self.messages:
                    if message["role"] == "user":
                        for item in message["content"]:
                            if item["type"] == "text":
                                gemini_content.append(item["text"])
                            elif item["type"] == "image_url":
                                # Decode base64 image and append PIL Image object
                                try:
                                    # Extract base64 string from data URL
                                    base64_str = item["image_url"]["url"].split(",")[1]
                                    image_data = base64.b64decode(base64_str)
                                    image = Image.open(io.BytesIO(image_data))
                                    gemini_content.append(image)
                                except Exception as e:
                                    print(f"Error decoding image for Gemini: {e}")
                                    continue
                    elif message["role"] == "assistant":
                        gemini_content.append(message["content"])

                response = self.gemini_model_vision.generate_content(gemini_content)
                agent_response = response.parts[0].text.strip()
            return agent_response
        except Exception as e:
            return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    load_dotenv()

    model_type = "openai" # Default to openai
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "gemini":
            model_type = "gemini"
        elif sys.argv[1].lower() == "openai":
            model_type = "openai"
        else:
            print("Usage: python3 -m radiology_agent [openai|gemini]")
            sys.exit(1)

    if model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            print("Error: OPENAI_API_KEY environment variable not set.")
            print("Please set the OPENAI_API_KEY environment variable or replace os.getenv(\"OPENAI_API_KEY\") with your actual key.")
            sys.exit(1)
    elif model_type == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            print("Error: GEMINI_API_KEY environment variable not set.")
            print("Please set the GEMINI_API_KEY environment variable.")
            sys.exit(1)

    case_path = "/home/bastien/RadioAgent/data/case1/non_contrast/"
    if not os.path.exists(case_path):
        print(f"Error: Case data path not found: {case_path}")
        print("Please ensure the directory '/home/bastien/RadioAgent/data/case1/non_contrast/' exists and contains images.")
        sys.exit(1)

    agent = RadiologyAgent(case_path, model_type=model_type)

    print(f"Initial slice: {agent.current_slice_index + 1}/{agent.max_slices}")

    # Simulate an initial query
    initial_text = "Patient presents with abdominal pain."
    initial_legend = "CT scan of the abdomen."
    initial_question = "What are your initial observations?"
    user_query = "Analyze this image and provide your observations. If you need to see more slices, finish your message with 'UP' if you want to move to the head or 'DOWN' for going to the toe. If you think you have a diagnosis, finish you message with 'DIAGNOSIS:' followed by your diagnosis"

    diagnosis_made = False
    while not diagnosis_made:
        # Construct the user's message for the current turn
        user_content = [
            {"type": "text", "text": f"{initial_text}\n\nLegend:{initial_legend}\n\n{initial_question}\n\n{user_query}"},
            *agent._get_current_image_content()
        ]
        
        # Append the user's message to the conversation history
        agent.messages.append({"role": "user", "content": user_content})

        print(f"\n--- Sending query for slice {agent.current_slice_index + 1} ---")
        agent_response = agent.get_llm_response()
        print(f"Agent Response: {agent_response}")

        # Append the agent's response to the conversation history
        agent.messages.append({"role": "assistant", "content": agent_response})

        if agent_response.upper().endswith("UP"):
            move_result = agent.move_slice("UP")
            print(move_result)
            user_query = "Okay, I've moved to the next slice. What are your observations now?"
        elif agent_response.upper().endswith("DOWN"):
            move_result = agent.move_slice("DOWN")
            print(move_result)
            user_query = "Okay, I've moved to the previous slice. What are your observations now?"
        elif agent_response.upper().startswith("DIAGNOSIS:"):
            print(f"Final Diagnosis: {agent_response}")
            diagnosis_made = True
        else:
            # If no special command, continue the conversation with the last response
            user_query = "Please continue your analysis or provide a diagnosis. If you need to see more slices, indicate 'UP' or 'DOWN'."
