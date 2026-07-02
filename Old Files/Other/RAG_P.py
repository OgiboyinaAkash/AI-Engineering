from groq import Groq
from sentence_transformers import SentenceTransformer
import os
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
def get_embeddings(text):
    return embedding_model.encode(text)


def llm_call(rag_prompt, api_key=""):
    if api_key == "":
        api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": rag_prompt
            }
        ],
        model="llama-3.3-70b-versatile"
    )
    return response.choices[0].message.content

database = [
    "Prabhat Restaurant has got very good Biryanis",
    "You can get very good coffee in Star Coffee.",
    "You should also try coffee at Nayan's cafe, it's much cheaper than Star Coffee and tastes exactly the same.",
    "Tara's restaurant has good North Indian food, but they don't serve non-vegetarian food",
    "You can eat a wide variety of Asian food at the Food Junction",
    "For vegetarians, you can try the Green Leaf Restaurant",
    "In evenings, you should definitely try the street food at Church Street",
    "Juice bar is a great place to get a juice, but there's no restaurant nearby",
]

def make_reqeust(user_prompt):
    cosine_similarities = []
    user_prompt_embedding = get_embeddings(user_prompt)
    for (data_idx, data) in enumerate(database):
        data_embedding = get_embeddings(data)
        cosine_similarities.append((data_idx, cosine_similarity(user_prompt_embedding, data_embedding)))
    cosine_similarities.sort(key = lambda x: x[1], reverse=True)


    similar_sentences = []
    for i in range(min(len(database), 3)):
        similar_sentences.append(database[cosine_similarities[i][0]])
    prompt = generate_prompt(similar_sentences, user_prompt)
    print(llm_call(prompt))

def generate_prompt(similar_sentences, user_prompt):
    return f"""
        You are an agent used for performing RAG. The data given to you is:
        {"\n\n\n\n".join(similar_sentences)}.



        The user has prompted:
        {user_prompt}
    """




def cosine_similarity(v1, v2):
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)
    return np.dot(v1, v2) / (magnitude_v1 * magnitude_v2)

make_reqeust("Where do I get a coffee?")
