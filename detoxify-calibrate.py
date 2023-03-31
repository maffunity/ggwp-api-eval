import time
import random
from collections import OrderedDict
import os
import glob
import csv
import sys
import sentence_cleaner
import math
import string
from detoxify import Detoxify
from urllib.parse import urlparse
from google.cloud import storage, bigquery

# future dataset from Google BigQuery
# class GCBDataset:

#     def __init__(gcb_table, sentence_cleaner=None, text_column_name='utterance', max_len=256):

#         if isinstance(gcb_table, str):
#             gcb_table = [gcb_table]

#         print ('Opening BigQuery client')
#         client = bigquery.Client()
#         texts = []
#         n_lines = 0
#         for table in gcb_table:
#             print ('Selecting {} from table {}'.format(column_name, table))
#             all_data = client.query("SELECT {} FROM `{}`".format(column_name, table)).to_dataframe()
#             for index, row in all_data.iterrows():
    
#                 text = row[column_name]
#                 if len(text)>max_len:
#                     continue

class CSVDataset:

    def __init__(self, dataset_path, sentence_cleaner=None):
        self.meta = OrderedDict()
        self.path = dataset_path
        self.sentence_cleaner = sentence_cleaner

        if dataset_path.startswith('gs:'):
            self.csvs = self.download_gcs(dataset_path, 'gcs_download')
        elif os.path.isdir(dataset_path):
            self.csvs = glob.glob(os.path.join(dataset_path,'*.csv'))
        elif isinstance(dataset_path,str):
            self.csvs = [dataset_path]

        self.data = self.load()

    def normalize_toxicity(self, toxicities):
        """
        Return whether toxic and a list of types of toxicity
        """
        benign_names = set(['benign','Benign','benign_gameplay']) 
        toxicities = [ tox.replace(' ','_') for tox in toxicities.split(',') ]
        toxicities = sorted([ t for t in toxicities if t not in benign_names ])
        toxic = any( False if t in benign_names else True for t in toxicities )
        return toxic, toxicities

    def get_data(self):
        return self.data

    def download_gcs(self, gs_url, local_dir=None):

        parsed = urlparse(gs_url)

        if parsed.scheme != "gs":
            return

        bucket_name = parsed.netloc
        path = parsed.path[1:] if parsed.path.startswith("/") else parsed.path

        out_files = []
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name=bucket_name)
        if local_dir is not None:
            # download as file or all contents of directory
            blobs = bucket.list_blobs(prefix=path)  # Get list of files
            for blob in blobs:
                blob_name = blob.name
                local_filename = os.path.join(local_dir, blob_name)
                out_files.append(local_filename)
                dir_filename = os.path.dirname(local_filename)
                print("   {}".format(blob_name))
                if not os.path.isdir(dir_filename):
                    os.makedirs(os.path.dirname(local_filename), exist_ok=True)
                blob.download_to_filename(local_filename)
            return out_files
        else:
            # iostream output, for single file only
            blob = storage.Blob(bucket=bucket, name=path)
            if blob.exists(storage_client):
                file_pointer = io.BytesIO(blob.download_as_string())
                return file_pointer
            else:
                return None

    def load(self):

        meta = OrderedDict()

        for csv_file in sorted(self.csvs):
            with open(csv_file, errors="ignore") as fp:
                csv_content = csv.reader(fp, delimiter=',', quotechar='"')
                line_no = 0
                header_found = False
                for row in csv_content:
                    if not header_found and line_no<=1: # look at first and possibly second line
                        uid_idx = None
                        text_idx = None
                        toxic_idx = {}
                        spkid_idx = None
                        audio_idx = None
                        for n, h in enumerate(row):
                            if h=='Sentence unique identifier':
                                uid_idx = n
                            elif h=='Sentence' or h=='automated_transcription':
                                text_idx = n
                            elif h=='Toxicity label':
                                # comma separated toxicity labels by voice actors
                                toxic_idx['toxicity'] = n
                            elif h=='benign_gameplay' or \
                                 h=='sexually explicit' or \
                                 h=='threat' or \
                                 h=='insult' or \
                                 h=='identity_comment':
                                # toxicity labels from customer data, by labellers
                                toxic_idx[h] = n
                            elif h=='Actor\'s unique Identifier' or h=='player_id':
                                spkid_idx = n
                            elif h=='Audio file path' or h=='track_url':
                                audio_idx = n
                        # import ipdb ; ipdb.set_trace()
                        if line_no>=1 and (text_idx is None or len(toxic_idx)==0 or audio_idx is None):
                            print ('could not find required fields in CSV file {}'.format(csv_file))
                            return
                        elif text_idx is not None and len(toxic_idx)>0 and audio_idx is not None:
                            print ('{}: uid_idx={}, text_idx={}, toxic_idx={}, spkid_idx={}, audio_idx={}'.format(csv_file, uid_idx, text_idx, toxic_idx, spkid_idx, audio_idx))
                            header_found = True
                        line_no += 1
                        continue
                    if uid_idx is None:
                        uid = os.path.splitext(os.path.basename(row[audio_idx]))[0]
                    else:
                        uid = row[uid_idx]
                    text = row[text_idx]
                    if self.sentence_cleaner is not None:
                        text = sentence_cleaner.clean_sentence(text) if sentence_cleaner is not None else text
                    if 'toxicity' in toxic_idx:
                        # voice actor labels
                        toxic = row[toxic_idx['toxicity']]
                    else:
                        # customer data labels
                        toxic = ','.join([ tox.replace(' ','_') for tox,tox_idx in toxic_idx.items() if row[tox_idx].lower()=='yes'])
                    if len(toxic)==0:
                        continue
                    spkid = row[spkid_idx]
                    if len(spkid)==0:
                        continue

                    line_no += 1
                    meta[uid] = {'spkid': spkid, 'text': text, 'toxicity': toxic}

        return meta


class Cache:

    def __init__(self, fn='cache'):
        self.fn = fn
        self.cache = OrderedDict()
        self.n_entries = 0
        if os.path.isfile (fn):
            self.load()
            self.fp = open (self.fn, 'at')
        else:
            self.fp = open (self.fn, 'wt')

    def add(self, text, result):
        self.cache[text] = result
        self.n_entries = len(self.cache)
        if self.fp is not None:
            self.fp.write('{}|{}\n'.format(text, '|'.join([ '{}={}'.format(key,result[key]) for key in sorted(result.keys()) ])))
            self.fp.flush()

    def get(self, text):
        if text in self.cache:
            ret = self.cache[text]
            return self.cache[text]
        else:
            return None
    
    def __contains__(self, text):
        return text in self.cache

    def load(self):
        with open(self.fn) as fp:
            for line in fp:
                line = line.strip()
                parts = line.split('|')
                result = {}
                if len(parts)>1:
                    text = parts[0]
                    for part in parts[1:]:
                        s = part.split('=')
                        if len(s)==2:
                            key = s[0]
                            value = float(s[1])
                            result[key] = value

                if len(result)>0:
                    self.cache[text] = result
            self.n_entries = len(self.cache)



class Calibration:

    def __init__(self, system_callback, exp_name, sys2gt_map=None, gt2sys_map=None, target_tol=0.01, max_egs=10, use_logits=False, timeout=300):
        self.system_callback = system_callback
        self.sys2gt_map = sys2gt_map
        self.gt2sys_map = gt2sys_map
        self.exp_name = exp_name
        self.target_tol = target_tol
        self.max_egs = max_egs
        self.use_logits = use_logits
        self.cache = None
        self.timeout = timeout

    def toxicity_map(self, toxicity_type, map=None):
        if map is None:
            return toxicity_type
        else:
            if toxicity_type in map:
                return map[toxicity_type]
            else:
                return None

    def get_toxicities(self, result, threshold=0.5, use_logits=False):
        """
        Return whether toxic and a list of types of toxicity
        """
        if isinstance(threshold, dict):
            thresholds = { tox:threshold[tox] if tox in threshold else 0.5 for tox in result }
        else:
            thresholds = { tox:threshold for tox in result }

        if use_logits:
            result = { k:math.log(v) for k,v in result.items() }

        toxicity = [ tox for tox,score in result.items() if tox in thresholds and result[tox]>thresholds[tox] ]
        toxicity = [ t.replace(' ','_') for t in toxicity ]
        return len(toxicity)>0, sorted(toxicity)

    def get_toxicities_in_gt_space(self, toxicity_types):
        """
        Return whether toxic and a list of types of toxicity
        """
        if self.sys2gt_map is not None:
            toxicity_types = [ self.sys2gt_map[tox] for tox in toxicity_types if tox in self.sys2gt_map ]
        toxicity = [ t.replace(' ','_') for t in toxicity_types ]
        return len(toxicity)>0, sorted(toxicity)

    def eval_system(self, dataset, th, use_logits):

        fps = OrderedDict()
        fns = OrderedDict()
        tns = OrderedDict()
        tps = OrderedDict()
        total = OrderedDict()
        human_label = True
        human_toxicity_types = OrderedDict( (self.toxicity_map(t, self.sys2gt_map),t) for t in th )
        for toxicity_type in human_toxicity_types:

            toxicity_type_str = '_'+toxicity_type if toxicity_type!='all' else ''
            human_label_str = '_hl1' if human_label else ''
            tp = 0 ; fp = 0 ; fn = 0 ; tn = 0 ; n_egs = 0
            if self.cache is None:
                print ('cache file {}'.format(os.path.join('stats',exp_name+'.cache')))
                self.cache = Cache(os.path.join('stats',exp_name+'.cache'))

            for n, (key, metadata) in enumerate(dataset.get_data().items()):
                text = metadata['text']
                gt_toxic, gt_toxicity_type = dataset.normalize_toxicity(metadata['toxicity'])
                if toxicity_type!='all':
                    gt_toxic = toxicity_type in gt_toxicity_type
                if text not in self.cache:
                    result = self.system_callback(text)
                    self.cache.add(text, result)
                else:
                    result = self.cache.get(text)
                predict_toxic, predict_toxicity_types = self.get_toxicities(result, threshold=th, use_logits=use_logits)
                
                # convert to human labels
                if human_label:
                    predict_toxic, predict_toxicity_types = self.get_toxicities_in_gt_space(predict_toxicity_types)
                # map tocixity_type to binary if evaluating a certain type of toxicity
                if toxicity_type!='all':
                    predict_toxic = toxicity_type in predict_toxicity_types

                tp += gt_toxic and predict_toxic
                fp += not gt_toxic and predict_toxic
                fn += gt_toxic and not predict_toxic
                tn += not gt_toxic and not predict_toxic
                n_egs += 1

                if self.max_egs is not None and n_egs>=self.max_egs:
                    break

            fp_rate = fp/n_egs
            fn_rate = fn/n_egs
            tp_rate = tp/n_egs
            tn_rate = tn/n_egs

            if toxicity_type!='all':
                detox_toxicity_type = self.toxicity_map(toxicity_type, self.gt2sys_map)
                tps[detox_toxicity_type] = tp_rate * 100
                tns[detox_toxicity_type] = tn_rate * 100
                fps[detox_toxicity_type] = fp_rate * 100
                fns[detox_toxicity_type] = fn_rate * 100
                total[detox_toxicity_type] = n_egs
        
        total_all = sum(total[detox_toxicity_type] for detox_toxicity_type in fps )
        total = {detox_toxicity_type:total[detox_toxicity_type]/total_all for detox_toxicity_type in fps }
        return fps, fns, tps, tns, total


    def calibrate(self, dataset, target_fps, target_tol=None, use_logits=None):

        target_tol = self.target_tol if target_tol is None else target_tol
        use_logits = self.use_logits if use_logits is None else use_logits

        ths = { toxicity_type: math.log(0.5) if use_logits else 0.5 for toxicity_type, th in target_fps.items() }
        ths_best = dict(ths)
        fps_best = { toxicity_type: 10000 for toxicity_type, th in ths.items() }
        fns_best = { toxicity_type: 10000 for toxicity_type, th in ths.items() }
        target_error_best = 10000
        target_errors_best = { toxicity_type: target_error_best for toxicity_type, th in ths.items() } 
        target_errors_best_last = { toxicity_type: target_error_best*2 for toxicity_type, th in ths.items() } 
        n_runs = 0
        opt_phase = 'global-random'
        learning_rate = { toxicity_type: 1.0 for toxicity_type, th in ths.items() }
        min_lr = 0.1
        grad_clip = 0.01
        cnt_no_improvement = 0
        cnt_no_improvements = { toxicity_type: 0 for toxicity_type, th in ths.items() }

        def error_fp(detox_toxicity_type, order=1.0):
            error = abs(fps[detox_toxicity_type] - target_fps[detox_toxicity_type])
            return pow(error, order)

        t0 = time.time()
        t_last_improvement = t0
        elapsed = 0.0
        timed_out = False

        while not timed_out and \
                any([ ( (fps_best[toxicity_type]>(target_fps[toxicity_type]+target_tol)) or \
                        (fps_best[toxicity_type]<(target_fps[toxicity_type]-target_tol)) ) for toxicity_type in ths ]):
            if opt_phase == 'global-random':
                    ths = { toxicity_type: math.log(random.random()) if use_logits else random.random() for toxicity_type, th in ths.items() }
            elif opt_phase == 'local-gdescent':
                # gradient descent for all toxicity categories
                for tox in target_errors.keys():
                    grad = learning_rate[tox] * (fps[tox]-target_fps[tox])/100.0
                    grad = max(-grad_clip,min(grad_clip, grad))
                    ths[tox] += grad
           
            # evaluate fp, fn rates for the current threshold
            fps, fns, tps, tns, total = self.eval_system(dataset=dataset, th=ths, use_logits=use_logits)

            target_errors = { detox_toxicity_type: (total[detox_toxicity_type]*error_fp(detox_toxicity_type) ) for detox_toxicity_type in fps }
            target_error = sum ( target_errors.values() )
            n_runs += 1

            cnt_no_improvements = { toxicity_type: (0 if target_errors[toxicity_type]<target_errors_best[toxicity_type] else cnt+1) for toxicity_type, cnt in cnt_no_improvements.items() }
            if target_error<target_error_best:
                target_error_best = target_error
                target_errors_best = dict(target_errors)
                ths_best = dict(ths)
                fps_best = dict(fps)
                fns_best = dict(fns)
                cnt_no_improvement = 0
            else:
                cnt_no_improvement += 1
                if opt_phase=='local-gdescent':
                    for toxicity_type in cnt_no_improvements:
                        if cnt_no_improvements[toxicity_type]>2:
                            cnt_no_improvements[toxicity_type] = 0
                            learning_rate[toxicity_type] *= 0.95
                            if learning_rate[toxicity_type]<min_lr:
                                learning_rate[toxicity_type]=min_lr
                if opt_phase=='global-random':
                    if cnt_no_improvement>50:
                        cnt_no_improvements = { toxicity_type: 0 for toxicity_type, cnt in cnt_no_improvements.items() }
                        opt_phase = 'local-gdescent'
                        ths = dict(ths_best)

            if n_runs % 10 == 0:
                print ('.', end='', file=sys.stdout)
                sys.stdout.flush()
                if any ( abs(target_errors_best[toxicity_type] - target_errors_best_last[toxicity_type])>0 for toxicity_type in target_errors_best ):
                    print ()
                    print ('opt-phase={}'.format(opt_phase))
                    for toxicity_type in fps_best:
                        within_tol = fps_best[toxicity_type]<=(target_fps[toxicity_type]+target_tol) and fps_best[toxicity_type]>=(target_fps[toxicity_type]-target_tol)
                        th_str = 'log(th)={:8.6f}'.format(ths_best[toxicity_type]) if use_logits else 'th={:8.6f}'.format(ths_best[toxicity_type])
                        print ('{:>15}: FP={:5.2f}% FN={:5.2f}% @ {}, within-tol:{:>3}, FP-target={:5.2f}±{:4.2}%, FP-error={:6.4f}, learning-rate={:6.3f}'.format(toxicity_type,fps_best[toxicity_type], fns_best[toxicity_type], th_str, 'yes' if within_tol else 'no', target_fps[toxicity_type], target_tol, target_errors_best[toxicity_type], learning_rate[toxicity_type]))
                    print ()
                    t_last_improvement = time.time()

                target_errors_best_last = dict(target_errors_best)

            elapsed = time.time() - t0
            timed_out = (time.time() - t_last_improvement) > self.timeout

        print ()
        print ('opt-phase={}'.format(opt_phase))
        for toxicity_type in fps_best:
            within_tol = fps_best[toxicity_type]<=(target_fps[toxicity_type]+target_tol) and fps_best[toxicity_type]>=(target_fps[toxicity_type]-target_tol)
            print ('{:>15}: FP={:5.2f}% FN={:5.2f}% @ th={:8.6f}, within-tol:{:>3}, FP-target={:5.2f}±{:4.2}%, FP-error={:6.4f}, learning-rate={:6.3f}'.format(toxicity_type,fps_best[toxicity_type], fns_best[toxicity_type], ths_best[toxicity_type], 'yes' if within_tol else 'no', target_fps[toxicity_type], target_tol, target_errors_best[toxicity_type], learning_rate[toxicity_type]))
        elapsed = time.time() - t0
        print ('time-elapsed={:.1f}s'.format(elapsed))
        if timed_out:
            print ('aborted due to timeout without optimization improvement')
        print ()
        return ths_best



if __name__=='__main__':
  
    sentence_cleaner = sentence_cleaner.SentenceCleaner(
        punctuation=False,
        uppercase=False,
        verbalize_numbers=False,
        verbalize_acronyms=True,
        remove_underscore=True,
        word_map_norm=os.path.join('assets','norm-word-map-llm-recognition.csv'),
        word_map_expand=os.path.join('assets','expand-word-map-llm-recognition.csv'),
    )
    norm_config_short_name = sentence_cleaner.get_short_config_str()
   
    # load CSV dataset, from 1+ local/GCS CSV file(s)
    # voice actor audio + human (voice actor) toxicity labelling
    # voice_actor_dataset_path='gs://unity-oto-ml-prd-us-central1-innodata/voice_acting/batch_1/outdated'
    # voice_actor_dataset = CSVDataset(voice_actor_dataset_path, sentence_cleaner)
    # gt2sys_map = {'Identity_comments':'identity_attack', 'Obscene':'obscene', 'Insult':'insult', 'Threat':'threat'}
    # sys2gt_map = {'identity_attack':'Identity_comments', 'obscene':'Obscene', 'insult':'Insult', 'threat': 'Threat'}
    # dataset = voice_actor_dataset

    # customer data GCP transcripts + human toxicity labelling
    customer_dataset_path = 'gs://unity-oto-ml-prd-us-central1-innodata/labelling/batch_20230310/labelling_batch_20230310_labels.csv'
    customer_dataset = CSVDataset(customer_dataset_path, sentence_cleaner)
    gt2sys_map = {'identity_comment':'identity_attack', 'sexually_explicit':'obscene', 'insult':'insult', 'threat':'threat'}
    sys2gt_map = {'identity_attack':'identity_comment', 'obscene':'sexually_explicit', 'insult':'insult', 'threat': 'threat'}
    dataset = customer_dataset

    # set random seed
    random.seed(777)
    # use human labels or GGWP labels for evaluation
    detoxify_model = Detoxify('original')
    # experiment name, to identify cache files
    exp_name = 'detoxify_eval_{}'.format(norm_config_short_name)
    # false-positive rates allowed after calibration
    target_false_positive_rates = {'obscene': 2.0, 'identity_attack': 2.0, 'insult': 2.0, 'threat': 2.0 }
    # start calibration
    calibration = Calibration(
            system_callback=detoxify_model.predict,
            exp_name=exp_name,
            sys2gt_map=sys2gt_map,
            gt2sys_map=gt2sys_map,
            target_tol=0.1,
            max_egs=5000,
            use_logits=False,
            timeout=30,
        )
    opt_thresholds = calibration.calibrate(dataset, target_false_positive_rates)
    print ('optimal thresholds: {}'.format(opt_thresholds))

