# Radiology Agent

This project implements a radiology agent that assists in medical image analysis and diagnosis using either OpenAI's o3 or Google's Gemini 2.5 Pro model. The agent can navigate through image slices and provide observations or a diagnosis based on the provided medical context and images.

## Setup

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository (if applicable):**
    ```bash
    # If this is a new project, you might skip this step or initialize git
    # git clone <your-repo-url>
    # cd RadiologyAgent
    ```

2.  **Install dependencies:**
    ```bash
    pip install openai python-dotenv Pillow google-generativeai
    ```

### API Key Configuration

The agent requires API keys for either OpenAI or Gemini, depending on which model you intend to use. It's recommended to store these keys in a `.env` file in the root directory of the project.

1.  Create a file named `.env` in the `/home/bastien/RadioAgent/` directory.
2.  Add your API keys to the `.env` file as follows:

    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    GEMINI_API_KEY="your_gemini_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` and `"your_gemini_api_key_here"` with your actual API keys.

## Usage

The agent can be run from the command line, specifying which AI model to use.

### Running with OpenAI (Default)

If no argument is provided, the agent will default to using OpenAI.
```bash
python3 -m radiology_agent
```

### Running with Gemini

To explicitly use the Gemini model:
```bash
python3 -m radiology_agent gemini
```

### Running with OpenAI (Explicit)

To explicitly use the OpenAI model:
```bash
python3 -m radiology_agent openai
```

### Interactive Session

Once the agent starts, it will begin an interactive session.
*   The agent will provide observations and analysis.
*   You can guide the agent by typing `UP` or `DOWN` (as part of your prompt or as the last word of the agent's response) to navigate through image slices.
*   When the agent is confident in a diagnosis, it will output `DIAGNOSIS:` followed by its conclusion.

## Project Structure

*   `radiology_agent.py`: The main script containing the `RadiologyAgent` class and the execution logic.
*   `data/`: Directory intended to hold medical case data, including image slices.
    *   `data/case1/non_contrast/`: Example directory for a specific case with image files (e.g., `1.jpg`, `2.jpg`, etc.).
*   `.env`: (Not committed to version control) Stores your API keys.

## Example Data Structure

For the agent to function correctly, your case data should be organized as follows:

```
/home/bastien/RadioAgent/
├── radiology_agent.py
├── .env
└── data/
    └── case1/
        └── non_contrast/
            ├── 1.jpg
            ├── 2.jpg
            └── ...
```

Ensure that the `case_path` variable in `radiology_agent.py` points to your actual image data directory.
