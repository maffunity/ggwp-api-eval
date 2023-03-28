from requests import post
from datetime import datetime
import base64
import json
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

voice_actor_dataset_path='/home/ext_marc_ferras_unity3d_com/project/corpora/unity-oto-ml-prd-us-central1-innodata/batch_1/innodata_results/'

def get_short_norm_config_name( punctuation, uppercase, verbalize_numbers, verbalize_acronyms, remove_underscore):

    punctuation = '1' if punctuation else '0'
    uppercase = '1' if uppercase else '0'
    verbalize_numbers = '1' if verbalize_numbers else '0'
    verbalize_acronyms = '1' if verbalize_acronyms else '0'
    remove_underscore = '1' if remove_underscore else '0'
    config_short_name = 'p{}u{}vn{}va{}ru{}'.format(
        punctuation,
        uppercase,
        verbalize_numbers,
        verbalize_acronyms,
        remove_underscore,
    )
    return config_short_name 


def detoxify_normalize_toxicity(result, threshold=0.5):
    """
    Return whether toxic and a list of types of toxicity
    """
    if isinstance(threshold, dict):
        threshold_obscene = threshold['obscene'] if 'obscene' in threshold else threshold
        threshold_identity_attack = threshold['identity_attack'] if 'identity_attack' in threshold else threshold
        threshold_insult = threshold['insult'] if 'insult' in threshold else threshold
        threshold_threat = threshold['threat'] if 'threat' in threshold else threshold
    else:
        threshold_obscene = threshold
        threshold_identity_attack = threshold
        threshold_insult = threshold
        threshold_threat = threshold

    toxicity = []
    if 'obscene' in result and result['obscene']>threshold_obscene:
        toxicity.append ('obscene')
    if 'identity_attack' in result and result['identity_attack']>threshold_identity_attack:
        toxicity.append ('identity_attack')
    if 'insult' in result and result['insult']>threshold_insult:
        toxicity.append ('insult')
    if 'threat' in result and result['threat']>threshold_threat:
        toxicity.append ('threat')
    toxicity = [ t.replace(' ','_') for t in toxicity ]
    return len(toxicity)>0, sorted(toxicity)


def detoxify_toxicity_to_human_label(predict_toxicity_type):
    """
    Return whether toxic and a list of types of toxicity
    """
    toxicity = []
    if 'obscene' in predict_toxicity_type:
        toxicity.append('Obscene')
    if 'insult' in predict_toxicity_type:
        toxicity.append('Insult')
    if 'identity_attack' in predict_toxicity_type:
        toxicity.append('Identity comments')
    if 'threat' in predict_toxicity_type:
        toxicity.append('Threat')
    toxicity = [ t.replace(' ','_') for t in toxicity ]
    return len(toxicity)>0, sorted(toxicity)

def voice_actor_normalize_toxicity(result, human_label=False):
    """
    Return whether toxic and a list of types of toxicity
    """
    result = result.split(',')
    toxicity = []
    if 'Benign' in result:
        if human_label:
            toxicity.append('Benign')
        else:
            toxicity.append('benign')
    if 'Identity comments' in result:
        if human_label:
            toxicity.append('Identity comments')
        else:
            toxicity.append('identity_hate')
    if 'Obscene' in result:
        if human_label:
            toxicity.append('Obscene')
        else:
            toxicity.append('sexual_content')
    if 'Insult' in result:
        if human_label:
            toxicity.append('Insult')
        else:
            toxicity.append('verbal abuse')
            toxicity.append('profanity')
    if 'Threat' in result:
        if human_label:
            toxicity.append('Threat')
        else:
            toxicity.append('violence')
            toxicity.append('self_harm')
    if 'Severe Toxicity' in result:
        if human_label:
            toxicity.append('Severe Toxicity')
        else:
            toxicity.append('violence')

    toxicity = [ t.replace(' ','_') for t in toxicity ]

    if human_label:
        return 'Benign' not in toxicity, sorted([ t for t in toxicity if t != 'Benign' ])
    else:
        return 'benign' not in toxicity, sorted([ t for t in toxicity if t != 'benign' ])

def get_voice_actor_dataset(csv_in, sentence_cleaner):
    meta = OrderedDict()
    if os.path.isdir(csv_in):
        csvs = glob.glob(os.path.join(csv_in,'*.csv'))
    else:
        csvs = [csv_in]
    for csv_file in csvs:
        print ('doing CSV file {}'.format(csv_file))
        with open(csv_file, errors="ignore") as fp:
            csv_file = csv.reader(fp, delimiter=',', quotechar='"')
            line_no = 0
            for row in csv_file:
                if line_no == 0:
                    uid_idx = None
                    text_idx = None
                    toxic_idx = None
                    spkid_idx = None
                    audio_idx = None
                    for n, h in enumerate(row):
                        if h=='Sentence unique identifier':
                            uid_idx = n
                        elif h=='Sentence':
                            text_idx = n
                        elif h=='Toxicity label':
                            toxic_idx = n
                        elif h=='Actor\'s unique Identifier':
                            spkid_idx = n
                        elif h=='Audio file path':
                            audio_idx = n
                    print ('uid_idx={}, text_idx={}, toxic_idx={}, spkid_idx={}, audio_idx={}'.format(uid_idx, text_idx, toxic_idx, spkid_idx, audio_idx))
                    if uid_idx is None or text_idx is None or toxic_idx is None or spkid_idx is None or audio_idx is None:
                        print ('could not find required fields in CSV file {}'.format(csv_file))
                        return
                    line_no += 1
                    continue
                if len(row)>=6:
                    uid = row[uid_idx]
                    text = row[text_idx]
                    text = sentence_cleaner.clean_sentence(text) if sentence_cleaner is not None else text
                    toxic = row[toxic_idx]
                    if len(toxic)==0:
                        continue
                    spkid = row[spkid_idx]
                    if len(spkid)==0:
                        continue
                    meta[uid] = {'spkid': spkid, 'text': text, 'toxicity': toxic}
            # break

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
            self.fp.write('{}|{}|{}|{}|{}\n'.format(text, result['identity_attack'], result['obscene'], result['insult'], result['threat']))
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
                if len(parts)==5:
                    text = parts[0]
                    result = {}
                    result['identity_attack'] = float(parts[1])
                    result['obscene'] = float(parts[2])
                    result['insult'] = float(parts[3])
                    result['threat'] = float(parts[4])
                    self.cache[text] = result
            self.n_entries = len(self.cache)


def eval_detoxify(th, human_toxicity_types, exp_name, log=False, system='detoxify', max_egs=5000):

    fps = OrderedDict()
    fns = OrderedDict()
    tns = OrderedDict()
    tps = OrderedDict()
    total = OrderedDict()
    human_label = True
    for toxicity_type in human_toxicity_types:
        if log:
            if len(toxicity_type)!='all':
                print ('\nevaluating toxicity type {} on system {}'.format(toxicity_type, system.upper()))
            else:
                print ('\nevaluating any toxicity on system {}'.format(system.upper()))

        toxicity_type_str = '_'+toxicity_type if toxicity_type!='all' else ''
        human_label_str = '_hl1' if human_label else ''
        tp = 0 ; fp = 0 ; fn = 0 ; tn = 0 ; n_egs = 0
        cache = Cache(os.path.join('stats',exp_name+'.cache'))
        for n, (key, metadata) in enumerate(voice_actor_dataset.items()):
            text = metadata['text']
            gt_toxic, gt_toxicity_type = voice_actor_normalize_toxicity(metadata['toxicity'], human_label=human_label)
            if toxicity_type!='all':
                gt_toxic = toxicity_type in gt_toxicity_type
            if text not in cache:
                result = detoxify_model.predict(text)
                cache.add(text, result)
            else:
                result = cache.get(text)
            predict_toxic, predict_toxicity_type = detoxify_normalize_toxicity(result, threshold=th)
            
            # use human labels or GGWP label
            if human_label:
                predict_toxic, predict_toxicity_type = detoxify_toxicity_to_human_label (predict_toxicity_type)
            # map tocixity_type to binary if evaluating a certain type of toxicity
            if toxicity_type!='all':
                predict_toxic = toxicity_type in predict_toxicity_type

            tp += gt_toxic and predict_toxic
            fp += not gt_toxic and predict_toxic
            fn += gt_toxic and not predict_toxic
            tn += not gt_toxic and not predict_toxic
            n_egs += 1

            if max_egs is not None and n_egs>=max_egs:
                break

        if log:
            print ('Toxicity type: {}'.format(toxicity_type))
            print ('Counts: tp={}, tn={}, fp={}, fn={}, total={}'.format(tp, tn, fp, fn, tp+tn+fp+fn))
        fp_rate = fp/n_egs
        fn_rate = fn/n_egs
        tp_rate = tp/n_egs
        tn_rate = tn/n_egs
        if log:
            print ('FP-rate: {:.1f}%, FN-rate: {:.1f}%'.format( fp_rate*100.0, fn_rate*100.0))
            print ('FP-rate: {:.1f}%, FN-rate: {:.1f}%'.format( fp_rate*100.0, fn_rate*100.0))

        if toxicity_type!='all':
            detox_toxicity_type = human2detoxify(toxicity_type)
            tps[detox_toxicity_type] = tp_rate * 100
            tns[detox_toxicity_type] = tn_rate * 100
            fps[detox_toxicity_type] = fp_rate * 100
            fns[detox_toxicity_type] = fn_rate * 100
            total[detox_toxicity_type] = n_egs
    
    total_all = sum(total[detox_toxicity_type] for detox_toxicity_type in fps )
    total = {detox_toxicity_type:total[detox_toxicity_type]/total_all for detox_toxicity_type in fps }
    return fps, fns, tps, tns, total

def human2detoxify(toxicity_type):
    if toxicity_type == 'Identity_comments':
        detox_toxicity_type = 'identity_attack'
    elif toxicity_type == 'Obscene':
        detox_toxicity_type = 'obscene'
    elif toxicity_type == 'Insult':
        detox_toxicity_type = 'insult'
    elif toxicity_type == 'Threat':
        detox_toxicity_type = 'threat'
    elif toxicity_type == 'all':
        detox_toxicity_type = 'all'
    return detox_toxicity_type

def detoxify2human(toxicity_type):
    if toxicity_type == 'identity_attack':
        detox_toxicity_type = 'Identity_comments'
    elif toxicity_type == 'obscene':
        detox_toxicity_type = 'Obscene'
    elif toxicity_type == 'insult':
        detox_toxicity_type = 'Insult'
    elif toxicity_type == 'threat':
        detox_toxicity_type = 'Threat'
    elif toxicity_type == 'all':
        detox_toxicity_type = 'all'
    return detox_toxicity_type


def calibrate(detoxify_max_fp, human_toxicity_types, eval_system_fn, exp_name, target_tol=0.01, max_egs=5000):

    ths = { toxicity_type: 0.5 for toxicity_type, th in detoxify_max_fp.items() }
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
        error = abs(fps[detox_toxicity_type] - detoxify_max_fp[detox_toxicity_type])
        return pow(error, order)

    t0 = time.time()
    while any([ ( (fps_best[toxicity_type]>(detoxify_max_fp[toxicity_type]+target_tol)) or \
                  (fps_best[toxicity_type]<(detoxify_max_fp[toxicity_type]-target_tol)) ) for toxicity_type in ths ]):
        if opt_phase == 'global-random':
                ths = { toxicity_type: random.random() for toxicity_type, th in ths.items() }
        elif opt_phase == 'local-gdescent':
            # gradient descent for all toxicity categories
            for tox in target_errors.keys():
                grad = learning_rate[tox] * (fps[tox]-detoxify_max_fp[tox])/100.0
                grad = max(-grad_clip,min(grad_clip, grad))
                ths[tox] += grad
       
        # evaluate fp, fn rates for the current threshold
        fps, fns, tps, tns, total = eval_system_fn(th=ths, human_toxicity_types=human_toxicity_types, exp_name=exp_name, log=False, max_egs=5000)

        target_errors = { detox_toxicity_type: (total[detox_toxicity_type]*error_fp(detox_toxicity_type) ) for detox_toxicity_type in fps }
        target_error = sum ( target_errors.values() )
        n_runs += 1

        cnt_no_improvements = { toxicity_type: (0 if target_errors[toxicity_type]<target_errors_best[toxicity_type] else cnt+1) for toxicity_type, cnt in cnt_no_improvements.items() }
        # print (cnt_no_improvement)
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
                    within_tol = fps_best[toxicity_type]<=(detoxify_max_fp[toxicity_type]+target_tol) and fps_best[toxicity_type]>=(detoxify_max_fp[toxicity_type]-target_tol)
                    print ('{:>15}: FP={:5.2f}% FN={:5.2f}% @ th={:8.6f}, within-tol:{:>3}, FP-target={:5.2f}±{:4.2}%, FP-error={:6.4f}, learning-rate={:6.3f}'.format(toxicity_type,fps_best[toxicity_type], fns_best[toxicity_type], ths_best[toxicity_type], 'yes' if within_tol else 'no', detoxify_max_fp[toxicity_type], target_tol, target_errors_best[toxicity_type], learning_rate[toxicity_type]))
                print ()

            target_errors_best_last = dict(target_errors_best)

    print ()
    print ('opt-phase={}'.format(opt_phase))
    for toxicity_type in fps_best:
        within_tol = fps_best[toxicity_type]<=(detoxify_max_fp[toxicity_type]+target_tol) and fps_best[toxicity_type]>=(detoxify_max_fp[toxicity_type]-target_tol)
        print ('{:>15}: FP={:5.2f}% FN={:5.2f}% @ th={:8.6f}, within-tol:{:>3}, FP-target={:5.2f}±{:4.2}%, FP-error={:6.4f}, learning-rate={:6.3f}'.format(toxicity_type,fps_best[toxicity_type], fns_best[toxicity_type], ths_best[toxicity_type], 'yes' if within_tol else 'no', detoxify_max_fp[toxicity_type], target_tol, target_errors_best[toxicity_type], learning_rate[toxicity_type]))
    elapsed = time.time() - t0
    print ('time-elapsed={:.1f}s'.format(elapsed))
    print ()
    return ths_best


if __name__=='__main__':
  
    punctuation=False
    uppercase=False
    verbalize_numbers=False
    verbalize_acronyms=True
    remove_underscore=True
    norm_config_short_name = get_short_norm_config_name( punctuation, uppercase, verbalize_numbers, verbalize_acronyms, remove_underscore)
    sentence_cleaner = sentence_cleaner.SentenceCleaner(
        punctuation=punctuation,
        uppercase=uppercase,
        verbalize_numbers=verbalize_numbers,
        verbalize_acronyms=verbalize_acronyms,
        remove_underscore=remove_underscore,
        word_map_norm=os.path.join('assets','norm-word-map-llm-recognition.csv'),
        word_map_expand=os.path.join('assets','expand-word-map-llm-recognition.csv'),
    )
    
    voice_actor_dataset = get_voice_actor_dataset(
            voice_actor_dataset_path,
            sentence_cleaner,
        )

    # set random seed
    random.seed(777)
    # number of sentences to eval in dataset
    max_egs = 5000
    # use human labels or GGWP labels for evaluation
    detoxify_model = Detoxify('original')
    # experiment name, to identify cache files
    exp_name = 'detoxify_eval_{}'.format(norm_config_short_name)
    human_toxicity_types = ['Identity_comments', 'Obscene', 'Insult', 'Threat']

    # maximum false-positive rates allowed
    # target_false_positive_rates = {'obscene': 10.0, 'identity_attack': 5.0, 'insult': 12.0, 'threat': 10.0 }
    target_false_positive_rates = {'obscene': 3.0, 'identity_attack': 1.0, 'insult': 5.0, 'threat': 4.0 }
    # target_false_positive_rates = {'obscene': 1.0, 'identity_attack': 1.0, 'insult': 1.0, 'threat': 1.0 }
    # start calibration
    opt_thresholds = calibrate(target_false_positive_rates, human_toxicity_types, eval_detoxify, exp_name, target_tol=0.01, max_egs=max_egs)
    print ('optimal thresholds: {}'.format(opt_thresholds))

