from hume import HumeBatchClient
from hume.models.config import FaceConfig
import pandas as pd
from sklearn.cluster import KMeans


client = HumeBatchClient("lGIslFXDOGG1Ofu1rqcybI1WZA9BVgqinVY90AAi0BmBfl0s")

class VideoDetection(object):
    def __init__(self, url):
        self.url = url

    def face_detection(self):
        temp1 = []
        for pred in self.job_predictions[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0][
            'predictions']:
            if pred['time'] >= 150 and pred['time'] <= 2648:
                temp1.append((pred['time'], pred['frame'], pred['box']))

        df = pd.DataFrame(columns=['time', 'frame', 'x', 'y', 'w', 'h', 'cluster'])
        for item in temp1:
            temp = [item[0], item[1], item[2]['x'], item[2]['y'], item[2]['w'], item[2]['h'], None]
            df.loc[len(df)] = temp

        model = KMeans(n_clusters=4)
        model.fit(df[['y']])
        df['cluster'] = model.predict(df[['y']])
        return df

    def bucket_maker(self, emotion_dict):
        happiness_enjoyment = ['Admiration', 'Adoration', 'Amusement', 'Awe', 'Excitement', 'Interest', 'Joy']
        anger = ['Anger', 'Contempt', 'Disgust']
        anxiety_confusion = ['Anxiety', 'Awkwardness', 'Distress', 'Doubt', 'Confusion']
        concentration = ['Concentration', 'Contemplation', 'Determination']
        sadness_disappointment = ['Sadness', 'Disappointment', 'Pain']
        boredom = ['Boredom', 'Tiredness']
        calmness = ['Calmness']
        realization_relief = ['Realization', 'Relief', 'Satisfaction']
        surprise = ['Surprise (negative)', 'Surprise (positive)']

        happiness_enjoyment_total = 0
        anger_total = 0
        anxiety_confusion_total = 0
        concentration_total = 0
        sadness_disappointment_total = 0
        boredom_total = 0
        calmness_total = 0
        realization_relief_total = 0
        surprise_total = 0
        for emotion in emotion_dict:
            if emotion['name'] in happiness_enjoyment:
                happiness_enjoyment_total += emotion['score']
            elif emotion['name'] in anger:
                anger_total += emotion['score']
            elif emotion['name'] in anxiety_confusion:
                anxiety_confusion_total += emotion['score']
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
            elif emotion['name'] in surprise:
                surprise_total += emotion['score']
        result = {}
        result['happiness_enjoyment'] = happiness_enjoyment_total / 7
        result['anger'] = anger_total / 3
        result['anxiety_confusion'] = anxiety_confusion_total / 5
        result['concentration'] = concentration_total / 3
        result['sadness_disappointment'] = sadness_disappointment_total / 3
        result['boredom'] = boredom_total / 2
        result['calmness'] = calmness_total / 1
        result['realization_relief'] = realization_relief_total / 3
        result['surprise'] = surprise_total / 2
        return result

    def build_dict(self, df):
        emotions_dict = dict()
        self.peaked_emotions_w_score_time_dict = dict()
        peak_threshold = 0.7

        for pred in self.job_predictions[0]['results']['predictions'][0]['models']['face']['grouped_predictions'][0][
            'predictions']:
            condition = (df['time'] == pred['time']) & (df['y'] == pred['box']['y']) & (
                        df['x'] == pred['box']['x']) & (df['w'] == pred['box']['w']) & (df['h'] == pred['box']['h'])
            row = df.loc[condition]
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
                    if cluster not in self.peaked_emotions_w_score_time_dict:
                        self.peaked_emotions_w_score_time_dict[cluster] = {}
                    if emotion not in self.peaked_emotions_w_score_time_dict[cluster]:
                        self.peaked_emotions_w_score_time_dict[cluster][emotion] = []
                    self.peaked_emotions_w_score_time_dict[cluster][emotion].append((buckets[emotion], pred['time']))

        self.emotions_average = dict()
        for cluster in emotions_dict:
            cluster_dict = emotions_dict[cluster]
            if cluster not in self.emotions_average:
                self.emotions_average[cluster] = {}
            total_score = 0
            for emotion in cluster_dict:
                total_score += cluster_dict[emotion]
            for emotion in cluster_dict:
                self.emotions_average[cluster][emotion] = cluster_dict[emotion] / total_score

    def print_output(self):
        result = ''
        for cluster in self.emotions_average:
            ascend_sorted_emotion_average = sorted(self.emotions_average[cluster].items(), key=lambda item: item[1],
                                                   reverse=True)
            result += 'The top expressed emotions for face {} are: '.format(cluster) + '\n'
            for i in range(0, 9):
                result += str((ascend_sorted_emotion_average[i])) + '\n'

        for cluster in self.peaked_emotions_w_score_time_dict:
            result += 'The emotions that peaked over 0.7' + ' : ' + 'for face ' + str(cluster) + '\n'
            for peaked_emotion, score_time_tuples in self.peaked_emotions_w_score_time_dict[cluster].items():
                result += peaked_emotion + ":"
                for tup in score_time_tuples:
                    result += "score of {} at {}".format(tup[0], tup[1]) + '\n'
        return result

    def generate_response(self):
        face_config = FaceConfig()
        job = client.submit_job(self.url, [face_config])
        result = job.await_complete()
        self.job_predictions = client.get_job_predictions(job_id=job.id)
        df = self.face_detection()
        self.build_dict(df)
        return self.print_output()