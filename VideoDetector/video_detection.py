from hume import HumeBatchClient
from hume.models.config import FaceConfig, ProsodyConfig
import pandas as pd
from sklearn.cluster import KMeans
import assemblyai as aai
import textstat
import nltk
from nltk.probability import FreqDist
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter, defaultdict
from fuzzywuzzy import process, fuzz

client = HumeBatchClient("lGIslFXDOGG1Ofu1rqcybI1WZA9BVgqinVY90AAi0BmBfl0s")
aai.settings.api_key = "48606b9d7b8e48dbb173f807512207d6"
nlp = spacy.load("en_core_web_sm")
class VideoDetection(object):
    def __init__(self, url, num_students):
        self.url = url
        self.num_students = num_students
        self.final_result = {}

    def bucket_maker(self, emotion_dict):
        happiness_enjoyment = ['Admiration', 'Adoration', 'Amusement', 'Awe', 'Excitement', 'Interest', 'Joy']
        anger = ['Anger', 'Contempt', 'Disgust']
        anxiety = ['Anxiety', 'Distress']
        confusion = ['Awkwardness', 'Doubt', 'Confusion']
        concentration = ['Concentration', 'Contemplation', 'Determination']
        sadness_disappointment = ['Sadness', 'Disappointment', 'Pain']
        boredom = ['Boredom', 'Tiredness']
        calmness = ['Calmness']
        realization_relief = ['Realization', 'Relief', 'Satisfaction']

        happiness_enjoyment_total = 0
        anger_total = 0
        anxiety_total = 0
        confusion_total = 0
        concentration_total = 0
        sadness_disappointment_total = 0
        boredom_total = 0
        calmness_total = 0
        realization_relief_total = 0
        for emotion in emotion_dict:
            if emotion['name'] in happiness_enjoyment:
                happiness_enjoyment_total += emotion['score']
            elif emotion['name'] in anger:
                anger_total += emotion['score']
            elif emotion['name'] in anxiety:
                anxiety_total += emotion['score']
            elif emotion['name'] in confusion:
                confusion_total += emotion['score']
            elif emotion['name'] in concentration:
                concentration_total += emotion['score']
            elif emotion['name'] in sadness_disappointment:
                sadness_disappointment_total += emotion['score']
            elif emotion['name'] in boredom:
                boredom_total += emotion['score']
            elif emotion['name'] in calmness:
                calmness_total += emotion['score']
            elif emotion['name'] in realization_relief:
                realization_relief_total += emotion['score']
        result = {}
        result['happiness_enjoyment'] = happiness_enjoyment_total / 7
        result['anger'] = anger_total / 3
        result['anxiety'] = anxiety_total / 2
        result['confusion'] = confusion_total / 3
        result['concentration'] = concentration_total / 3
        result['sadness_disappointment'] = sadness_disappointment_total / 3
        result['boredom'] = boredom_total / 2
        result['calmness'] = calmness_total / 1
        result['realization_relief'] = realization_relief_total / 3
        return result
    def video_analysis(self):
        face_config = FaceConfig()
        urls = [self.url]
        job = client.submit_job(urls, [face_config])
        result = job.await_complete(timeout=3600)
        job_predictions = client.get_job_predictions(job_id=job.id)
        start = job_predictions[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][0]['time'] + 600
        end = job_predictions[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions'][-1]['time'] - 600

        temp1 = []
        for pred in job_predictions[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0]['predictions']:
            if pred['time'] >= start and pred['time'] <= end:
                temp1.append((pred['time'], pred['frame'], pred['box']))

        df = pd.DataFrame(columns=['time', 'frame', 'x', 'y', 'w', 'h', 'cluster'])
        for item in temp1:
            temp = [item[0], item[1], item[2]['x'], item[2]['y'], item[2]['w'], item[2]['h'], None]
            df.loc[len(df)] = temp

        model = KMeans(n_clusters=self.num_students + 1)
        model.fit(df[['y']])
        df['cluster'] = model.predict(df[['y']])
        cluster_means = df.groupby('cluster')['y'].mean().reset_index()
        cluster_means = cluster_means.sort_values(by='y', ascending=True).reset_index(drop=True)
        cluster_mapping = {row['cluster']: i for i, row in cluster_means.iterrows()}

        df['cluster'] = df['cluster'].map(cluster_mapping)

        # The start and end time range of predictions to be processed
        start_time = start
        end_time = end

        # A threshold of what is defined as a peaked emotion
        peak_threshold = 0.6

        emotions_dict = dict()
        peaked_emotions_w_score_time_dict = dict()

        for pred in job_predictions[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0][
            'predictions']:
            if pred['time'] >= start_time and pred['time'] <= end_time:
                condition = (df['time'] == pred['time']) & (df['y'] == pred['box']['y']) & (
                            df['x'] == pred['box']['x']) & (df['w'] == pred['box']['w']) & (df['h'] == pred['box']['h'])
                row = df.loc[condition]
                if len(row) > 0:
                    row = row.iloc[0]
                    cluster = int(row['cluster'])
                    if cluster not in emotions_dict:
                        emotions_dict[cluster] = {}
                    cluster_dict = emotions_dict[cluster]
                    buckets = self.bucket_maker(pred['emotions'])
                    for emotion in buckets.keys():
                        if emotion not in cluster_dict:
                            cluster_dict[emotion] = buckets[emotion]
                        else:
                            cluster_dict[emotion] = cluster_dict[emotion] + buckets[emotion]
                        if buckets[emotion] >= peak_threshold:
                            if cluster not in peaked_emotions_w_score_time_dict:
                                peaked_emotions_w_score_time_dict[cluster] = {}
                            if emotion != 'calmness':
                                if emotion not in peaked_emotions_w_score_time_dict[cluster]:
                                    peaked_emotions_w_score_time_dict[cluster][emotion] = []
                                peaked_emotions_w_score_time_dict[cluster][emotion].append((buckets[emotion], pred['time']))

        emotions_average = dict()

        for cluster in emotions_dict:
            cluster_dict = emotions_dict[cluster]
            if cluster not in emotions_average:
                emotions_average[cluster] = {}
            total_score = 0
            for emotion in cluster_dict:
                total_score += cluster_dict[emotion]
            for emotion in cluster_dict:
                emotions_average[cluster][emotion] = cluster_dict[emotion] / total_score

        self.final_result['video_analysis'] = {}
        self.final_result['video_analysis']['avg_emotions'] = {}
        avg_emotion_result = self.final_result['video_analysis']['avg_emotions']
        for cluster in emotions_average:
            ascend_sorted_emotion_average = sorted(emotions_average[cluster].items(), key=lambda item: item[1],
                                                   reverse=True)
            avg_emotion_result[cluster] = ascend_sorted_emotion_average

        self.final_result['video_analysis']['peak_emotions'] = {}
        peak_emotion_result = self.final_result['video_analysis']['peak_emotions']
        for cluster in peaked_emotions_w_score_time_dict:
            peak_emotion_result[cluster] = {}
            for peaked_emotion, score_time_tuples in peaked_emotions_w_score_time_dict[cluster].items():
                peak_emotion_result[cluster][peaked_emotion] = []
                for tup in score_time_tuples:
                    temp = {"Score" : tup[0], "Time" : tup[1]}
                    peak_emotion_result[cluster][peaked_emotion].append(temp)

    def find_assembly_match(self,assembly_data, hume_item, start_index=0):
        if len(assembly_data) == 0:
            return None
        mid = len(assembly_data) // 2
        assembly_start = assembly_data[mid][1]
        assembly_end = assembly_data[mid][2]
        if assembly_start <= hume_item['time']['begin'] and assembly_end >= hume_item['time']['end']:
            return start_index + mid
        elif hume_item['time']['begin'] < assembly_start:
            return self.find_assembly_match(assembly_data[:mid], hume_item, start_index)
        else:
            return self.find_assembly_match(assembly_data[mid + 1:], hume_item, start_index + mid + 1)

    def get_wordnet_pos(self,tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def group_similar_words(self, word_freq):
        grouped_words = defaultdict(int)
        processed_words = set()

        for item in word_freq:
            lst = item.split()
            for word in lst:
                if word in processed_words:
                    continue
                # Find similar words
                similar_words = process.extractBests(word, word_freq.keys(), scorer=fuzz.token_sort_ratio,
                                                     score_cutoff=60)
                # Aggregate frequencies
                total_freq = sum(word_freq[w] for w, score in similar_words)
                grouped_words[word] = total_freq
                processed_words.update(w for w, score in similar_words)

        return grouped_words

    def audio_analysis(self):
        prosody_config = ProsodyConfig()
        urls = [self.url]
        job = client.submit_job(urls, [prosody_config])
        result = job.await_complete(timeout=3600)
        hume_data = client.get_job_predictions(job_id=job.id)
        hume_data = hume_data[0]['results']['predictions'][0]['models']['prosody']['grouped_predictions'][0]['predictions']


        config = aai.TranscriptionConfig(
            speech_model=aai.SpeechModel.best,
            speaker_labels=True,
            speakers_expected=self.num_students + 1,
            language_code="en_us"
        )

        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(self.url)
        speaker_time = {}
        teacher_transcript = []
        for utterance in transcript.utterances:
            if utterance.speaker not in speaker_time:
                speaker_time[utterance.speaker] = 0
            speaker_time[utterance.speaker] += utterance.end - utterance.start
        maxTime = 0
        maxSpeaker = ''
        for speaker, time in speaker_time.items():
            if time > maxTime:
                maxTime = time
                maxSpeaker = speaker
        for utterance in transcript.utterances:
            if utterance.speaker == maxSpeaker:
                temp = [utterance.text, utterance.start / 1000, utterance.end / 1000]
                teacher_transcript.append(temp)

        emotions_dict = {}
        for item in hume_data:
            assembly_idx = self.find_assembly_match(teacher_transcript, item)
            if assembly_idx is not None:
                buckets = self.bucket_maker(item['emotions'])
                for emotion in buckets:
                    if emotion not in emotions_dict:
                        emotions_dict[emotion] = 0
                    emotions_dict[emotion] += buckets[emotion]

        emotions_avg = {}
        total = 0
        for emotion in emotions_dict:
            total += emotions_dict[emotion]
        for emotion in emotions_dict:
            emotions_avg[emotion] = emotions_dict[emotion] / total

        self.final_result['audio_analysis'] = {}
        self.final_result['audio_analysis']['teacher_avg_emotions'] = []
        ascend_sorted_emotion_average = sorted(emotions_avg.items(), key=lambda item: item[1], reverse=True)
        for i in range(0, 9):
            self.final_result['audio_analysis']['teacher_avg_emotions'].append(ascend_sorted_emotion_average[i])

        teacherTime = 0
        totalStart = transcript.utterances[0].start / 1000
        totalEnd = transcript.utterances[-1].end / 1000
        for line in teacher_transcript:
            teacherTime += line[2] - line[1]
        self.final_result['audio_analysis']['teacher_talk_time_percent'] = teacherTime / (totalEnd - totalStart)

        teacherTime = 0
        count = 0
        for line in teacher_transcript:
            teacherTime += line[2] - line[1]
            count += 1
        self.final_result['audio_analysis']['avg_teacher_talk_time(s)'] = teacherTime / count

        totalScore = 0
        count = 0
        for text in teacher_transcript[1:]:
            count += 1
            totalScore += textstat.dale_chall_readability_score(text[0])
        self.final_result['audio_analysis']['teacher_readability_score'] = totalScore / count

        text = ''
        for line in teacher_transcript[1:]:
            text += line[0]
        questions = [sent for sent in nltk.sent_tokenize(text) if '?' in sent]
        self.final_result['audio_analysis']['num_questions_asked_by_teacher'] = len(questions)

        words = nltk.word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        words = [word.lower() for word in words if word.isalpha() and word.lower() not in stop_words]

        lemmatizer = WordNetLemmatizer()
        pos_tags = nltk.pos_tag(words)
        lemmatized_words = [lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)) for word, tag in pos_tags]
        fdist = FreqDist(lemmatized_words)
        common_words = fdist.most_common(10)
        self.final_result['audio_analysis']['teacher_common_words'] = common_words

        doc = nlp(text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        name_counts = Counter(names)

        grouped_word_freq = self.group_similar_words(name_counts)
        self.final_result['audio_analysis']['teacher_common_names'] = grouped_word_freq

    def generate_response(self):
        self.video_analysis()
        self.audio_analysis()
        return self.final_result