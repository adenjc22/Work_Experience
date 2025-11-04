# Voice-Based Customer Service AI Assistant

A closed-loop AI system that provides intelligent voice-based customer service with persistent memory, conversation logging, and automatic ticket generation.

## Features

- **Voice Interaction**: Hold spacebar to record voice queries, receive spoken responses
- **Persistent Memory**: Uses ChromaDB to remember past customer interactions across sessions
- **SAP Integration**: Fetches customer information from SAP database (Excel format)
- **Conversation Logging**: Automatically saves all interactions in structured JSON logs
- **Ticket Generation**: Creates support tickets with AI-generated summaries at conversation end
- **Audio Transcription**: Powered by OpenAI Whisper for accurate speech-to-text
- **Text-to-Speech**: AI-generated voice responses with fallback to local TTS

## Project Structure

```
Work_Experience/
├── main.py                 # Main application entry point
├── number_gen.py           # Utility for generating test phone numbers
├── requirements.txt        # Python dependencies
├── demo_sap.xlsx          # SAP customer data (not included in repo)
├── .env                   # API keys (not included in repo)
├── chroma_db/             # Persistent vector database for memory
├── logs/                  # Conversation logs (JSON format)
└── tickets/               # Generated support tickets (JSON format)
```

## Prerequisites

- Python 3.8+
- OpenAI API key
- Microphone and speakers/headphones
- Windows OS (uses PowerShell)

## Installation

1. Clone the repository:
```powershell
git clone https://github.com/adenjc22/Work_Experience.git
cd Work_Experience
```

2. Install dependencies:
```powershell
pip install -r requirements.txt
```

3. Create a `.env` file in the project root:
```
API_KEY=your_openai_api_key_here
```

4. Prepare your SAP data:
   - Create `demo_sap.xlsx` with columns: `CustomerPhone`, and other customer info fields

## Usage

Run the main application:
```powershell
python main.py
```

### Controls

- **Hold Spacebar**: Record your voice input
- **Release Spacebar**: Stop recording and process input
- **Say "goodbye"**: End conversation gracefully (generates ticket)
- **Say "quit"**: End conversation immediately (generates ticket)
- **Ctrl+C**: Emergency exit (generates ticket)

### How It Works

1. **Customer Identification**: Enter or specify customer phone number
2. **Voice Input**: Hold spacebar and speak your query
3. **AI Processing**: 
   - Transcribes audio using Whisper
   - Retrieves past conversation memories from ChromaDB
   - Fetches customer data from SAP
   - Generates contextual response using GPT-4
4. **Response**: AI speaks the reply (TTS)
5. **Memory Storage**: Interaction is stored in vector database for future reference
6. **Logging**: Conversation is saved to JSON log file
7. **Ticket Generation**: On conversation end, AI generates structured support ticket

## Key Components

### Memory System
- Uses **ChromaDB** for semantic similarity search
- Stores customer interactions as vector embeddings
- Retrieves relevant past conversations to maintain context

### Conversation Logs
- JSON format with session ID and timestamp
- Tracks all inputs, outputs, and phone numbers
- Marks conversations as resolved/unresolved

### Support Tickets
- Auto-generated summaries using GPT-4
- Includes: ticket ID, customer phone, issue summary, resolution, next actions, status
- Stored as individual JSON files

## Configuration

Edit these variables in `main.py`:

```python
SAMPLE_RATE = 16000      # Audio sample rate (Hz)
CHANNELS = 1             # Mono audio
THRESHOLD = 0.02         # Volume threshold
CHUNK_SIZE = 1024        # Audio chunk size
```

## Models Used

- **Whisper**: `small` model for transcription
- **GPT-4o-mini**: For conversation and ticket generation
- **text-embedding-3-small**: For vector embeddings
- **gpt-4o-mini-tts**: For text-to-speech (with pyttsx3 fallback)

## Example Workflow

```
1. Customer calls: "Hi, I need help with my order"
2. AI retrieves past interactions and SAP data
3. AI responds: "Hello! I see you ordered item X. How can I help?"
4. Conversation continues...
5. Customer says: "goodbye"
6. System generates ticket with summary and resolution notes
```

## Limitations

- Windows-only (spacebar detection uses `keyboard` library)
- Requires active internet connection for OpenAI API
- SAP data must be in Excel format with specific column names

## Future Enhancements

- [ ] Multi-language support
- [ ] Web dashboard for ticket management
- [ ] Real-time sentiment analysis

## License

This project is for educational/demonstration purposes.

## Author

Aden Campbell

## Acknowledgments

- OpenAI for Whisper and GPT models
- ChromaDB for vector database
- Community contributors
