import os
from sklearn.metrics.pairwise import pairwise_distances_argmin

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from utils import *


class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'], binary = True)
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name[0] + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)#### YOUR CODE HERE ####
        """
        question_vec = np.reshape(question_vec,(1,self.embeddings_dim))
        sim_list = []
        for ind, can_emb in zip(thread_ids, thread_embeddings):
            sim = cosine_similarity(question_vec,can_emb)
            sim_list.append(sim[0][0])
        
        
        sort_ind = np.argsort(sim_list)[::-1]
        """
        print(thread_embeddings.shape)
        print(thread_ids.shape)

        best_thread = pairwise_distances_argmin(thread_embeddings, question_vec, axis=0)[0]#sort_ind[0] #### YOUR CODE HERE ####
        print(best_thread)
        return thread_ids[best_thread]


class DialogueManager(object):
    paths = dict({'INTENT_RECOGNIZER': ".."
            ,'TFIDF_VECTORIZER': ".."
            ,'TAG_CLASSIFIER': ".."
            ,'WORD_EMBEDDINGS': ".."
            ,'THREAD_EMBEDDINGS_FOLDER':"thread_embeddings_by_tags"})
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        chatbot = ChatBot(
                'Ron Obvious',
                trainer='chatterbot.trainers.ChatterBotCorpusTrainer'
        )

        # Create a new trainer for the chatbot
        trainer = ChatterBotCorpusTrainer(chatbot)

        # Train based on english greetings corpus
        trainer.train("chatterbot.corpus.english.greetings")

        # Train based on the english conversations corpus
        trainer.train("chatterbot.corpus.english.conversations")

        # Train the chatbot based on the english corpus
        trainer.train("chatterbot.corpus.english")


        self.chatbot = chatbot

       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)#### YOUR CODE HERE ####
        features = self.tfidf_vectorizer.transform(prepared_question)#### YOUR CODE HERE ####
        intent = self.intent_recognizer.predict(features) #### YOUR CODE HERE ####

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(question)#### YOUR CODE HERE ####
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)#### YOUR CODE HERE ####
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(question, tag)#### YOUR CODE HERE ####
            print(thread_id)
            return self.ANSWER_TEMPLATE % (tag[0], thread_id)