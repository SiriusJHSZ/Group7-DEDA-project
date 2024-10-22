import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from nltk import ngrams
import seaborn as sns
import os
import numpy as np
from gensim import corpora
from gensim.models import LdaMulticore, TfidfModel
from gensim.models.coherencemodel import CoherenceModel
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS as GENSIM_STOPWORDS
from gensim.models.phrases import Phrases, Phraser
import pyLDAvis
import pyLDAvis.gensim_models
from nltk.stem import WordNetLemmatizer
import nltk
from multiprocessing import cpu_count


# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def remove_boilerplate(text):
    # List of boilerplate phrases and patterns to remove
    boilerplate_patterns = [
        r'Equal Opportunity Employer.*?(?=\n|\Z)',
        r'All qualified applicants.*?(?=\n|\Z)',
        r'without regard to.*?(?=\n|\Z)',
        r'will receive consideration.*?(?=\n|\Z)',
        r'EEO Employer.*?(?=\n|\Z)',
        r'gender identity.*?(?=\n|\Z)',
        r'veteran status.*?(?=\n|\Z)',
        r'sexual orientation.*?(?=\n|\Z)',
        r'race, color, religion.*?(?=\n|\Z)',
        r'company is committed to.*?(?=\n|\Z)',
        r'For more information.*?(?=\n|\Z)',
        r'pay transparency.*?(?=\n|\Z)',
        r'we are an equal opportunity employer.*?(?=\n|\Z)',
        r'this job description.*?(?=\n|\Z)',
        r'compensation provided obtaining.*?(?=\n|\Z)',
        r'provided obtaining rating.*?(?=\n|\Z)',
       
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    return text

def process_text(text):
    # Remove boilerplate text
    text = remove_boilerplate(text)
    
    # Remove HTML tags and special characters
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    
    # Define stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    additional_stops = set([
        'experience', 'skills', 'ability', 'work', 'team', 'job', 'position', 'role', 'candidate',
        'required', 'requirements', 'qualifications', 'year', 'years', 'including', 'must',
        'will', 'us', 'excellent', 'knowledge', 'looking', 'join', 'company', 'new',
        'provide', 'data', 'analysis', 'analyst', 'responsibilities', 'working', 'provide',
        'one', 'using', 'development', 'support', 'environment', 'also', 'customer',
        'may', 'application', 'information', 'related', 'business', 'management',
        'supporting', 'help', 'ensure', 'across', 'solutions', 'day', 'daytoday',
        'technologies', 'needs', 'product', 'products', 'tasks',
        'etc', 'per', 'week', 'report', 'reports', 'reporting',
        'equal', 'opportunity', 'employer', 'regard', 'race', 'color', 'religion',
        'sex', 'national', 'origin', 'disability', 'status', 'gender', 'identity',
        'orientation', 'veteran', 'applicants', 'consideration', 'employment',
        'receive', 'without', 'state', 'local', 'laws', 'law', 'applicable',
        'additional', 'duties', 'assigned', 'perform', 'includes', 'limited', 'other',
        'e', 'g', 'etc', 'us', 'new', 'may', 'get', 'much', 'would', 'could',
        'compensation', 'provided', 'obtaining', 'rating', 'k',
        'full', 'time', 'benefits', 'package', 'comprehensive',
        # Add more as needed
    ])
    stop_words = stop_words.union(additional_stops)
    
    # Simple tokenization and stop word removal
    tokens = simple_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(14, 7))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud of Job Descriptions")
    plt.show()

def analyze_ngrams(text, n):
    tokens = simple_tokenize(text)
    n_grams = ngrams(tokens, n)
    return Counter(n_grams).most_common(20)

def analyze_tfidf(text):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.toarray()[0]
    return sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)[:20]

def extract_entities(text, chunk_size=500000):
    nlp = spacy.load("en_core_web_sm")
    entities = []
    text_length = len(text)
    for i in range(0, text_length, chunk_size):
        chunk = text[i:i+chunk_size]
        doc = nlp(chunk)
        entities.extend([(ent.text, ent.label_) for ent in doc.ents])
    return entities

def extract_skills(text):
    skills = {
        "python": r'\bpython\b',
        "sql": r'\bsql\b',
        "excel": r'\bexcel\b',
        "data analysis": r'\bdata analysis\b',
        "machine learning": r'\bmachine learning\b',
        "statistics": r'\bstatistics\b',
        "data visualization": r'\bdata visualization\b',
        "r": r'\br\b',
        "tableau": r'\btableau\b',
        "power bi": r'\bpower bi\b',
        "hadoop": r'\bhadoop\b',
        "spark": r'\bspark\b',
        "aws": r'\baws\b',
        "azure": r'\bazure\b',
        "communication": r'\bcommunication\b',
        "problem-solving": r'\bproblem[- ]solving\b',
        "critical thinking": r'\bcritical thinking\b',
        "data science": r'\bdata science\b',
        "big data": r'\bbig data\b',
        "deep learning": r'\bdeep learning\b',
        "java": r'\bjava\b',
        "c++": r'\bc\+\+\b',
        "javascript": r'\bjavascript\b',
        "nosql": r'\bnosql\b',
        "mongodb": r'\bmongodb\b',
        "tensorflow": r'\btensorflow\b',
        "pytorch": r'\bpytorch\b',
        "certified analytics professional": r'\bcertified analytics professional\b',
        "microsoft certified": r'\bmicrosoft certified\b',
    }

    skill_counts = {skill: len(re.findall(pattern, text, re.IGNORECASE)) for skill, pattern in skills.items()}
    return {k: v for k, v in skill_counts.items() if v > 0}

def visualize_skills(skills):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(skills.values()), y=list(skills.keys()), orient='h')
    plt.title("Skills Mentioned in Job Descriptions")
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.show()

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def compute_coherence_values(dictionary, corpus, texts, start, limit, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics,
                             workers=cpu_count()-1, chunksize=2000, passes=10, alpha='asymmetric')
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


# Main execution
if __name__ == "__main__":
    # Read the text from the file
    file_path = '/Users/flewolf/Downloads/descriptions.txt'
    if not os.path.exists(file_path):
        print(f"The file {file_path} does not exist. Please check the path.")
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()

        # Process the text
        processed_text = process_text(text)

        # Generate and display word cloud
        generate_word_cloud(processed_text)

        # Analyze bigrams
        print("Top 20 Bigrams:")
        bigrams = analyze_ngrams(processed_text, 2)
        for gram, count in bigrams:
            print(f"{' '.join(gram)}: {count}")

        # Analyze trigrams
        print("\nTop 20 Trigrams:")
        trigrams = analyze_ngrams(processed_text, 3)
        for gram, count in trigrams:
            print(f"{' '.join(gram)}: {count}")

        # TF-IDF analysis
        print("\nTop 20 TF-IDF terms:")
        tfidf_terms = analyze_tfidf(processed_text)
        for term, score in tfidf_terms:
            print(f"{term}: {score}")

        # Named Entity Recognition
        print("\nNamed Entities:")
        entities = extract_entities(text)
        entity_counter = Counter(entities)
        for (entity, label), count in entity_counter.most_common(20):
            print(f"{entity} ({label}): {count}")

        # Skill extraction and visualization
        skills = extract_skills(processed_text)
        print("\nSkills mentioned:")
        for skill, count in skills.items():
            print(f"{skill}: {count}")
        visualize_skills(skills)

    
