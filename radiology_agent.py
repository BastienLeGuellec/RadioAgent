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
import argparse

class RadiologyAgent:
    def __init__(self, case_data_path, model_type="openai"):
        self.model_type = model_type
        if self.model_type == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_type == "gemini":
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.gemini_model_text = genai.GenerativeModel("gemini-pro")
            self.gemini_model_vision = genai.GenerativeModel("gemini-pro-vision")
        else:
            raise ValueError("Invalid model_type. Must be 'openai' or 'gemini'.")

        self.base_case_path = case_data_path
        self.phases = self._get_available_phases()
        self.current_phase = None
        self.image_paths = []
        self.current_slice_index = 0
        self.max_slices = 0
        self.messages = [
            {"role": "system", "content": f"You are a helpful radiology assistant. You can respond with 'UP' or 'DOWN' as the last word of your message to request a slice change, or 'DIAGNOSIS:' followed by your diagnosis. To switch phase, say the phase name in capital letters."}
        ]

    def _get_available_phases(self):
        return [d for d in os.listdir(self.base_case_path) if os.path.isdir(os.path.join(self.base_case_path, d))]

    def initialize_phase(self, phase_name):
        if phase_name in self.phases:
            self.current_phase = phase_name
            self.image_paths = self._load_image_paths(self.current_phase)
            if not self.image_paths:
                return f"Error: No images found in phase '{phase_name}'."
            self.current_slice_index = 0
            self.max_slices = len(self.image_paths)
            self.messages[0]['content'] = f"You are a helpful radiology assistant. You can respond with 'UP' or 'DOWN' as the last word of your message to request a slice change, or 'DIAGNOSIS:' followed by your diagnosis. Available phases are: {', '.join(self.phases)}. To switch phase, say the phase name in capital letters. The current phase is {self.current_phase}."
            return f"Initialized {phase_name} phase. Initial slice: {self.current_slice_index + 1}/{self.max_slices}"
        else:
            return f"Invalid phase. Available phases are: {', '.join(self.phases)}"

    def _load_image_paths(self, phase):
        phase_path = os.path.join(self.base_case_path, phase)
        image_patterns = [os.path.join(phase_path, "*.jpg"), os.path.join(phase_path, "*.png")]
        paths = []
        for pattern in image_patterns:
            paths.extend(glob.glob(pattern))
        paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        return paths

    def switch_phase(self, phase_name):
        if phase_name in self.phases:
            self.current_phase = phase_name
            self.image_paths = self._load_image_paths(self.current_phase)
            if not self.image_paths:
                return f"Error: No images found in phase '{phase_name}'."
            self.current_slice_index = 0
            self.max_slices = len(self.image_paths)
            return f"Switched to {phase_name} phase. Initial slice: {self.current_slice_index + 1}/{self.max_slices}"
        else:
            return f"Invalid phase. Available phases are: {', '.join(self.phases)}"

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
            image_extension = os.path.splitext(current_image_path)[1].lower()
            mime_type = "image/jpeg" if image_extension == ".jpg" else "image/png"
            return [
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded_image}"}}
            ]
        return []

    def move_slice(self, direction):
        if direction.upper() == "DOWN":
            if self.current_slice_index < self.max_slices - 1:
                self.current_slice_index += 1
                return f"Moved to slice {self.current_slice_index + 1}/{self.max_slices}. Current image updated."
            else:
                return f"Already at the last slice ({self.current_slice_index + 1}/{self.max_slices}). Cannot move further DOWN."
        elif direction.upper() == "UP":
            if self.current_slice_index > 0:
                self.current_slice_index -= 1
                return f"Moved to slice {self.current_slice_index + 1}/{self.max_slices}. Current image updated."
            else:
                return f"Already at the first slice ({self.current_slice_index + 1}/{self.max_slices}). Cannot move further UP."
        else:
            return "Invalid direction. Use 'UP' or 'DOWN'."

    def get_llm_response(self, text_only=False):
        try:
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model="o3", 
                    messages=self.messages,
                )
                agent_response = response.choices[0].message.content.strip()
            elif self.model_type == "gemini":
                gemini_content = []
                for message in self.messages:
                    if message["role"] == "user":
                        content_list = message["content"] if isinstance(message["content"], list) else [message["content"]]
                        for item in content_list:
                            if isinstance(item, dict) and item.get("type") == "text":
                                gemini_content.append(item["text"])
                            elif not text_only and isinstance(item, dict) and item.get("type") == "image_url":
                                try:
                                    base64_str = item["image_url"]["url"].split(",")[1]
                                    image_data = base64.b64decode(base64_str)
                                    image = Image.open(io.BytesIO(image_data))
                                    gemini_content.append(image)
                                except Exception as e:
                                    print(f"Error decoding image for Gemini: {e}")
                                    continue
                            elif isinstance(item, str):
                                gemini_content.append(item)
                    elif message["role"] == "assistant":
                        gemini_content.append(message["content"])
                
                model_to_use = self.gemini_model_text if text_only else self.gemini_model_vision
                response = model_to_use.generate_content(gemini_content)
                agent_response = response.parts[0].text.strip()
            return agent_response
        except Exception as e:
            return f"An unexpected error occurred: {e}"

if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Radiology Agent CLI")
    parser.add_argument("model_type", choices=["openai", "gemini"], help="The type of model to use.")
    parser.add_argument("case_id", help="The ID of the case to load (e.g., case1).")
    args = parser.parse_args()

    if args.model_type == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            print("Error: OPENAI_API_KEY environment variable not set.")
            sys.exit(1)
    elif args.model_type == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key is None:
            print("Error: GEMINI_API_KEY environment variable not set.")
            sys.exit(1)

    case_path = f"/home/bastien/RadioAgent/data/{args.case_id}/"
    if not os.path.exists(case_path):
        print(f"Error: Case data path not found: {case_path}")
        sys.exit(1)

    agent = RadiologyAgent(case_path, model_type=args.model_type)

    # Initial phase selection
    initial_text = "Patient presents with abdominal pain."
    initial_legend = "CT scan of the abdomen."
    initial_question = f"Please select a contrast phase to begin the analysis. The available phases are: {', '.join(agent.phases)}"
    
    agent.messages.append({"role": "user", "content": f"{initial_text}\n\nLegend:{initial_legend}\n\n{initial_question}"})

    print("\n--- Requesting initial phase selection ---")
    agent_response = agent.get_llm_response(text_only=True)
    print(f"Agent response: {agent_response}")
    agent.messages.append({"role": "assistant", "content": agent_response})

    selected_phase = None
    for phase in agent.phases:
        if phase in agent_response.lower():
            selected_phase = phase
            break

    if selected_phase:
        initialization_result = agent.initialize_phase(selected_phase)
        print(initialization_result)
    else:
        print("Could not determine a phase from the agent's response. Exiting.")
        sys.exit(1)

    if not agent.current_phase or not agent.image_paths:
        print("Could not initialize a valid phase with images. Exiting.")
        sys.exit(1)

    print(f"Initial phase: {agent.current_phase}. Initial slice: {agent.current_slice_index + 1}/{agent.max_slices}")

    user_query = f"Analyze this image and provide your observations. If you need to see more slices, finish your message with 'UP' if you want to move to the head or 'DOWN' for going to the toe. If you think you have a diagnosis, finish you message with 'DIAGNOSIS:' followed by your diagnosis. Available phases are: {', '.join(agent.phases)}. To switch phase, say the phase name in capital letters."

    diagnosis_made = False
    while not diagnosis_made:
        user_content = [
            {"type": "text", "text": user_query},
            *agent._get_current_image_content()
        ]
        
        agent.messages.append({"role": "user", "content": user_content})

        print(f"\n--- Sending query for slice {agent.current_slice_index + 1} in {agent.current_phase} phase ---")
        agent_response = agent.get_llm_response()
        print(f"Agent Response: {agent_response}")

        agent.messages.append({"role": "assistant", "content": agent_response})

        phase_switched = False
        for phase in agent.phases:
            if agent_response.upper().endswith(phase):
                print(agent.switch_phase(phase))
                user_query = "Okay, I've switched to the new phase. What are your observations now?"
                phase_switched = True
                break
        
        if phase_switched:
            continue

        if agent_response.upper().endswith("UP"):
            move_result = agent.move_slice("UP")
            print(move_result)
            user_query = move_result
        elif agent_response.upper().endswith("DOWN"):
            move_result = agent.move_slice("DOWN")
            print(move_result)
            user_query = move_result
        elif agent_response.upper().startswith("DIAGNOSIS:"):
            print(f"Final Diagnosis: {agent_response}")
            diagnosis_made = True
        else:
            user_query = "Please continue your analysis or provide a diagnosis. If you need to see more slices, indicate 'UP' or 'DOWN'."
