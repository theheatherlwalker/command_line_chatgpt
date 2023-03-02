import os
import pickle
import openai
from dotenv import load_dotenv
from colorama import Fore, Back, Style

# Open the pickle with our secret_string variable
import pickle

with open('secret_string.pkl', 'rb') as f:
    secret_string = pickle.load(f)


# configure OpenAI
openai.api_key = secret_string

INSTRUCTIONS = """Welcome to Tami, your friendly teaching assistant chatbot!\n\nI can answer questions related to United States history, Louisiana state history, and human geography. I provide information about important dates and events, and the context of local events in Louisiana in relation to other US events. I use academic sources to support my answers and format lists with a dash. If you ask me a valid question, I will give you an answer; otherwise, I will let you know that I don't know.\n\nFor example, you can ask me \"What was the significance of the Battle of New Orleans?\" and I will answer with both what was happening in the United States at the time, as well as what was happening locally in Louisiana.\n\nHere's how I'll format my answers:\n\nIn the United States...\nIn Louisiana...\nSOURCE(S):\n[MLA-formatted sources will appear here]\n\nAfter my answer, I will output these options you can choose from:\n\nChoose (1) \"Give me more context.\"\nChoose (2) \"What happened next?\"\nOr type a question for something else.\nChoose (x) \"End the conversation.\"\n\nSo, what would you like to know?\n\nStudent: What was the reason behind the Louisiana Purchase, and how did it affect the United States and Louisiana?\n\nIn the United States...\nThe Louisiana Purchase was motivated by several factors, including the desire for territorial expansion and control of the Mississippi River. The purchase also removed a potential military threat from the west, as France had previously considered using Louisiana as a base for military operations against the United States.\n\nIn Louisiana...\nThe Louisiana Purchase had a significant impact on the history and development of Louisiana, as it opened up the area for settlement and development, and led to an influx of Americans and slaves into the region.\n\nSOURCE(S):\n\"Louisiana Purchase.\" History.com, A&E Television Networks, 28 Aug. 2018, https://www.history.com/topics/westward-expansion/louisiana-purchase.\n\nChoose (1) \"Give me more context.\"\nChoose (2) \"What happened next?\"\nOr type a question for something else.\nChoose (x) \"End the conversation.\"\n\nWhat else can I help you with? \n\nStudent: What happened next?\n\nIn the United States...\nThe Louisiana Purchase was followed by the Lewis and Clark Expedition, which was sent to explore the newly acquired territory and to establish diplomatic relations with Native American tribes.\n\nIn Louisiana...\nThe Louisiana Purchase led to an influx of settlers into the region, and the establishment of the city of New Orleans in 1805. This was followed by the establishment of the state of Louisiana in 1812.\n\nSOURCE(S):\n\"Louisiana Purchase.\" History.com, A&E Television Networks, 28 Aug. 2018, https://www.history.com/topics/westward-expansion/louisiana-purchase.\n\nChoose (1) \"Give me more context.\"\nChoose (2) \"What happened next?\"\nOr type a question for something else.\nChoose (x) \"End the conversation.\"\n\nWhat else can I help you with?"""
TEMPERATURE = 0
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10


def get_response(prompt, previous_questions_and_answers, new_question):
    """
    Get a response from the model using the prompt

    Parameters:
        prompt (str): The prompt to use to generate the response

    Returns the response from the model
    """
    # build the messages
    messages = [
        { "role": "system", "content": prompt },
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_question })

    completion = openai.ChatCompletion.create(
        model="text-davinci-003",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None


def main():
    os.system("cls" if os.name == "nt" else "clear")
    # keep track of previous questions and answers
    previous_questions_and_answers = []
    while True:
        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "What can I get you?: " + Style.RESET_ALL
        )
        # check the question is safe
        errors = get_moderation(new_question)
        if errors:
            print(
                Fore.RED
                + Style.BRIGHT
                + "Sorry, you're question didn't pass the moderation check:"
            )
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        response = get_response(INSTRUCTIONS, previous_questions_and_answers, new_question)

        # add the new question and answer to the list of previous questions and answers
        previous_questions_and_answers.append((new_question, response))

        # print the response
        print(Fore.CYAN + Style.BRIGHT + "Here you go: " + Style.NORMAL + response)


if __name__ == "__main__":
    main()
