# Jarvis: Your Conversational AI Companion 🤖

Welcome to Jarvis – your friendly neighborhood chatbot project!

Jarvis is designed to make technology more accessible, engaging, and even a little fun. Whether you’re here to ask questions, learn something new, or simply have a laugh, Jarvis is ready to chat with you.

This project combines cutting-edge AI with a touch of humor and personality to deliver an interactive experience you’ll love. It is a Python-based chatbot capable of responding to user queries using TF-IDF for natural language processing and cosine similarity for response matching. It also includes a predefined set of responses for common questions.

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

## Integrating External APIs: Weather Feature (Beware this part is a little more technical)

* In this project, We can extend this simple chatbot's functionality further by integrating an external APIs. This Weather API feature integration enables the chatbot to provide real-time weather updates for any location upon request.

* Choose a Weather API:
  * Sign up for a free or paid weather API like OpenWeatherMap or WeatherAPI.
  * Obtain your API key after creating an account.

* Replace the weather API Code with your own code in the WeatherAPI class in Jarvis.py file:
```shell
class WeatherAPI:
    def __init__(self, weather_api_key):
        self.weather_api_key = weather_api_key

    def get_weather(self, query_param):
        api_host = "open-weather13.p.rapidapi.com"
        base_url = f"https://open-weather13.p.rapidapi.com/city/{query_param}/EN"
        
        headers = {
            "X-RapidAPI-Key": self.weather_api_key,
            "X-RapidAPI-Host": api_host
        }

        try:
            # Making the request to the API
            response = requests.get(base_url, headers=headers)

            # Check if the request was successful (HTTP status code 200)
            if response.status_code != 200:
                return f"Error: Unable to fetch data. Status code: {response.status_code}"

            data = response.json()

            # Check if the response contains error information
            if 'error' in data:
                return f"Error: {data['error']['message']}"

            # Extracting relevant weather data
            location = data.get('location', {}).get('name', 'Unknown')
            country = data.get('location', {}).get('country', 'Unknown')
            condition_text = data.get('current', {}).get('condition', {}).get('text', 'No data')
            temperature_c = data.get('current', {}).get('temp_c', 'No data')
            humidity = data.get('current', {}).get('humidity', 'No data')

            # Return the weather summary
            return f"The weather in {location}, {country} is {condition_text}. " \
                   f"The temperature is {temperature_c}°C, and the humidity is {humidity}%. "

        except requests.RequestException as e:
            return f"Error fetching weather information: {e}"  
```

* now we can update Chat method to add this feature
```shell
elif weather_api_key!="your_weather_api_goes_here" and "weather" in user_response:
      print("Sure! Let me fetch the weather details for you.")
      city = user_response.split("in")[-1].strip()
      # Ensure the city name is not empty
      if city:
          weather_info = weather_api.get_weather(city)
          print(f"Jarvis: {weather_info}")
      else:
          print("Jarvis: I couldn't understand the city name. Could you please provide it after 'in'?")
```

### Adding Similar Features
The process for integrating additional external APIs or features is similar:

* Identify the Feature: Decide on the functionality you'd like to add (e.g., stock prices, news updates, currency conversion).

* Choose an API: Find a reliable API for the desired feature, and sign up to get an API key.

* Implement the Functionality:

* Write a function/class in Jarvis.py to fetch and process data from the API.
* Handle errors gracefully in case of invalid inputs or connectivity issues.
* Integrate Into the Chatbot:
  * Update the chatbot's logic to recognize user queries related to the new feature.
  * Call the respective API function and format the response appropriately.
  * Test Thoroughly: Ensure the integration works as expected for various inputs.

## Customizing Jarvis
Want to make Jarvis even cooler? Check out the Additional Responses section in the code to add your own custom replies to specific inputs.

## Contributing: 
Feedback and contributions are always welcome! 🎉

If you’d like to:
* Report a bug
* Suggest a feature
* Improve the code

Feel free to open an issue or submit a pull request. Let’s make Jarvis the best chatbot ever, together!

Enjoy your chatbot interactions!

## Show Your Support
If you love Jarvis, give this project a ⭐ on GitHub!

## Ready to Chat?
Let’s get started! Fire up Jarvis and see what he has to say.
Happy chatting! 🚀

