# Jarvis: Your Conversational AI Companion 🤖

Welcome to Jarvis – your friendly neighborhood chatbot project!

Jarvis is designed to make technology more accessible, engaging, and even a little fun. Whether you’re here to ask questions, learn something new, or simply have a laugh, Jarvis is ready to chat with you.

The motivation behind Jarvis is simple: I wanted to create a simple and conversational AI that’s not only smart but also approachable. This project combines cutting-edge AI with a touch of humor and personality to deliver an interactive experience you’ll love.

So, whether you’re a tech enthusiast, a curious learner, or just passing by, I hope Jarvis will make your day a little brighter.

## Features

It is a Python-based chatbot capable of responding to user queries using TF-IDF for natural language processing and cosine similarity for response matching. It also includes a predefined set of responses for common questions.

Features
* Modular Design: Encapsulated in a Chatbot class for better organization and scalability.
* Dynamic Conversations: Uses TF-IDF and cosine similarity to match user queries with responses from a text corpus.
* Predefined Responses: Handles common questions directly with predefined answers.
* Greeting Detection: Responds to greetings with friendly messages.
* Exit Commands: Allows users to exit the chat by typing "bye" or showing gratitude ("thanks"/"thank you").

## Getting Started

These instructions will guide you through setting up and running the chatbot project on your local machine.

### Prerequisites

Before you begin, make sure you have the following installed on your system:

- Python3.X: [Download Python](https://www.python.org/downloads/)
- Required libraries:
  numpy
  nltk
  sklearn

### Installation

1. Clone or download this project to your local machine.

2. Open your terminal or command prompt and navigate to the project directory using the `cd` command:

   ```shell
   cd path/to/chatbot-project
* Create a virtual environment (optional but recommended):

```
python -m venv venv
```

Activate the virtual environment:

On Windows:
```
venv\Scripts\activate
```
On macOS and Linux:

```
source venv/bin/activate
```

* Install the required libraries and dependencies using pip:

```
pip install -r requirements.txt
```
* Usage
Once everything is set up, let’s get Jarvis talking!

Make sure you are still in the project directory and have your virtual environment activated.

Start Jarvis by running:

```
python jarvis.py
```
Jarvis will greet you with a warm welcome. From there, you can:
Ask questions
Engage in conversations
Say "bye" to end the session

## How It Works
Corpus Processing: The chatbot reads and preprocesses a text file (chatbot.txt) to build a corpus for response matching.
TF-IDF Vectorization: Converts sentences into numerical vectors for similarity comparison.
Predefined Responses: A dictionary handles common questions like "Tell me a joke" or "What's the capital of Australia."
Class-based Design: All chatbot functionalities are encapsulated in a Chatbot class.

## Example Interaction

```
Jarvis: My name is Jarvis. I'm here to answer your questions. Type 'bye' to exit.
You: hi
Jarvis: hello
You: Who created you 
Jarvis: It was Nishant, the wizard of code, who single-handedly brought me to life!
You: bye
Jarvis: Goodbye! Feel free to chat again anytime!
```

* ## Customizing Jarvis
Want to make Jarvis even cooler? Check out the Additional Responses section in the code to add your own custom replies to specific inputs.

* ## Contributing: 
Feedback and contributions are always welcome! 🎉

If you’d like to:
* Report a bug
* Suggest a feature
* Improve the code

Feel free to open an issue or submit a pull request. Let’s make Jarvis the best chatbot ever, together!

Enjoy your chatbot interactions!

* ## Show Your Support
If you love Jarvis, give this project a ⭐ on GitHub!

* ## Ready to Chat?
Let’s get started! Fire up Jarvis and see what he has to say.
Happy chatting! 🚀

