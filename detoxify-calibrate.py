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

api_key = '3_Hb2c23mrmtsJSIvbODo2aditTDh7Wt5Linsdmgq-rY1aKvrbS5sdaru9iorralt9OotavWoeSq2qaqgHqAgnt1vG0'
base_url = 'https://api.ggwp.com/chat/v2/message'
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


def calibration_converged():
    return False

def eval_system(th, log=False):

    # print ('th(identity_attack)={:.4f}, th(obscene)={:.4f}, th(insult)={:.4f}, th_threat={:.4f}'.format(
    #     th['identity_attack'],
    #     th['obscene'],
    #     th['insult'],
    #     th['threat'],
    #     ))
    fps = OrderedDict()
    fns = OrderedDict()
    tns = OrderedDict()
    tps = OrderedDict()
    total = OrderedDict()
    for toxicity_type in toxicity_types:
        if log:
            if len(toxicity_type)!='all':
                print ('\nevaluating toxicity type {} on system {}'.format(toxicity_type, system.upper()))
            else:
                print ('\nevaluating any toxicity on system {}'.format(system.upper()))

        toxicity_type_str = '_'+toxicity_type if toxicity_type!='all' else ''
        human_label_str = '_hl1' if human_label else ''
        # eval
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
                if system =='detoxify':
                    predict_toxic, predict_toxicity_type = detoxify_toxicity_to_human_label (predict_toxicity_type)
            # map tocixity_type to binary if evaluating a certain type of toxicity
            if toxicity_type!='all':
                predict_toxic = toxicity_type in predict_toxicity_type

            # print ('checking',text, gt_toxicity_type, predict_toxicity_type)
            tp += gt_toxic and predict_toxic
            fp += not gt_toxic and predict_toxic
            fn += gt_toxic and not predict_toxic
            tn += not gt_toxic and not predict_toxic
            n_egs += 1
            if gt_toxic != predict_toxic:
                if gt_toxic and not predict_toxic:
                    if not per_toxicity_type:
                        print('\'{}\': FALSE_NEGATIVE, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
                else:
                    if not per_toxicity_type:
                        print('\'{}\': FALSE-POSITIVE ERROR, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
            else:
                if gt_toxic and predict_toxic:
                    if not per_toxicity_type:
                        print('\'{}\': TRUE-POSITIVE, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
                else:
                    if not per_toxicity_type:
                        print('\'{}\': TRUE-NEGATIVE, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))

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
            tps[detox_toxicity_type] = tp_rate
            tns[detox_toxicity_type] = tn_rate
            fps[detox_toxicity_type] = fp_rate
            fns[detox_toxicity_type] = fn_rate
            total[detox_toxicity_type] = n_egs
            # target_error += abs(fp_rate - detoxify_max_fp[detox_toxicity_type]) + 
            # target_error += max(0,fp_rate - detoxify_max_fp[detox_toxicity_type]) + max(0,fn_rate - detoxify_max_fn[detox_toxicity_type])
            # target_error+= abs(fp_rate - detoxify_max_fp[detox_toxicity_type]) + max(0,fn_rate - detoxify_max_fn[detox_toxicity_type])
    
    total_all = sum(total[detox_toxicity_type] for detox_toxicity_type in fps )
    total = {detox_toxicity_type:total[detox_toxicity_type]/total_all for detox_toxicity_type in fps }
    return fps, fns, tps, tns, total

# def update_thresholds (fps, fns, ths, fps_last, fns_last, ths_last):
def update_thresholds (ths):

    new_ths = { toxicity_type: random.random() for toxicity_type, th in ths.items() }
    
    # if fps_last is None or fns_last is None or ths_last is None:
    #     # no past data, make a change randomly
    #     new_ths = { toxicity_type: min(1.0,max(0,th*(1+random.gauss(0.0,0.4)))) for toxicity_type, th in ths.items() }
    # else:
    #     new_ths = dict(fps)
    #     for detox_toxicity_type in fps:
    #         toxicity_type = detoxify2human(detox_toxicity_type)
    #         if detox_toxicity_type!='all':
    #             # print (fps[detox_toxicity_type])
    #             # print (fps_last[detox_toxicity_type])
    #             # print (ths[detox_toxicity_type])
    #             # print (ths_last[detox_toxicity_type])
    #             if (ths[detox_toxicity_type] - ths_last[detox_toxicity_type]) != 0.0:
    #                 slope_fp = (fps[detox_toxicity_type] - fps_last[detox_toxicity_type]) / (ths[detox_toxicity_type] - ths_last[detox_toxicity_type])
    #                 if slope_fp != 0.0:
    #                     print ('slope[{}]={:.5f}'.format(detox_toxicity_type, slope_fp))
    #                     new_th = ths[detox_toxicity_type] - 0.1 * slope_fp
    #                     # if slope_fp!=0.0:
    #                     #     # fp_last[toxicity_type] + slope_fp * (th -th_last) == 0.02
    #                     #     new_th = (detoxify_max_fp[detox_toxicity_type] - fps_last[detox_toxicity_type] + slope_fp*ths_last[detox_toxicity_type] ) / slope_fp
    #                     # else:
    #                     #     new_th = ths[detox_toxicity_type]
    #                 else:
    #                     print ('no gradient change')
    #                     new_th = min(1.0,max(0,ths[detox_toxicity_type]*(1+random.gauss(0.0,0.4))))
    #             else:
    #                 print ('no threshold change')
    #                 new_th = min(1.0,max(0,ths[detox_toxicity_type]*(1+random.gauss(0.0,0.4))))
    #             new_ths[detox_toxicity_type] = new_th
    return new_ths

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



    max_egs = 5000
    # use human labels or GGWP labels for evaluation
    human_label = True
    # break-down evaluation per toxicity type
    per_toxicity_type = True
    system='detoxify'
    # detoxify_threshold=0.8
    # detoxify_threshold = {'obscene': 0.9898, 'identity_attack': 0.057, 'insult': 0.9525 , 'threat': 0.78 }
    detoxify_threshold = {'obscene': 0.5, 'identity_attack': 0.5, 'insult': 0.5 , 'threat': 0.5 }
    detoxify_max_fp = {'obscene': 0.02, 'identity_attack': 0.02, 'insult': 0.02, 'threat': 0.02 }
    detoxify_max_fn = {'obscene': 0.12, 'identity_attack': 0.12, 'insult': 0.12, 'threat': 0.12 }
    if system=='detoxify':
        from detoxify import Detoxify
        detoxify_model = Detoxify('original')
    else:
        detoxify_model = None
        config_short_name = get_short_config_name(config)

    exp_name = '{}_eval_{}'.format(
                system,
                norm_config_short_name)

    if per_toxicity_type:
        if human_label:
            toxicity_types = ['all','Identity_comments', 'Obscene', 'Insult', 'Threat']
        else:
            toxicity_types = ['all','verbal_abuse', 'profanity', 'sexual_content', 'identity_hate', 'self_harm']
    else:
        toxicity_types = ['all']

    # initial system run with equal thresholds
    # detoxify_threshold = {'obscene': 0.5, 'identity_attack': 0.5, 'insult': 0.5 , 'threat': 0.5 }
    # ths_last = dict(detoxify_threshold)
    # fps_last, fns_last = eval_system(th=ths_last, log=False)
    # for toxicity_type in fps_last:
    #     print ('{}: {:4f}% @ th={:.4f}'.format(toxicity_type,fps_last[toxicity_type]*100.0, ths_last[toxicity_type]))
    # ths = update_thresholds (fps_last, fns_last, ths_last, None, None, None)
    # print ()
    target_error_best = 10000
    ths = dict(detoxify_threshold)
    ths_best = dict(ths)
    fps_best = dict(detoxify_threshold)
    fns_best = dict(detoxify_threshold)
    cnt_no_improvement = 0
    order_fp = 1.0
    order_fn = 1.0
    n_runs = 0
    opt_phase = 'global'
    weight_local_linear = 1.0
    while target_error_best>0:
        if opt_phase == 'global':
                ths = { toxicity_type: random.random() for toxicity_type, th in ths.items() }
        elif opt_phase == 'local-linear':
           # random category
           l = list(target_errors)
           random.shuffle(l)
           tox = l[0]
           # get target and current FP rate 0<fp<1
           target = detoxify_max_fp[tox]
           current = fps[tox]
           if current>target:
               # we want fewer FPs => raise threshold
               grad = weight_local_linear * (current-target)
               grad = min(0.01, grad)
               ths[tox] += grad
           else:
               # we can lower the threshold a bit to allow more FPs
               grad = weight_local_linear * (target-current)
               grad = min(0.01, grad)
               ths[tox] -= grad
           # print ('toxicity', tox, 'current', current, 'target', target, 'diff', current-target,'gradient', (current-target), 'grad',grad)
       
        # evaluate fp, fn rates for the current threshold
        fps, fns, tps, tns, total = eval_system(th=ths, log=False)

        def error_fp(detox_toxicity_type, order):
            error = abs(fps[detox_toxicity_type] - detoxify_max_fp[detox_toxicity_type]) * 100.0
            return pow(error, order)
        target_errors = { detox_toxicity_type: (total[detox_toxicity_type]*error_fp(detox_toxicity_type,order_fp) ) for detox_toxicity_type in fps }
        target_error = sum ( target_errors.values() )
        n_runs += 1

        if target_error<target_error_best:
            target_error_best = target_error
            target_errors_best = dict(target_errors)
            ths_best = dict(ths)
            fps_best = dict(fps)
            fns_best = dict(fns)
            cnt_no_improvement = 0
        else:
            cnt_no_improvement += 1
            if opt_phase=='local-linear' and cnt_no_improvement>5:
                cnt_no_improvement = 0
                weight_local_linear *= 0.95
                if weight_local_linear<0.1:
                    weight_local_linear=0.1
            if opt_phase=='global' and cnt_no_improvement>50:
                cnt_no_improvement = 0
                opt_phase = 'local-linear'
                # start with best threshold so far
                ths = dict(ths_best)

        if n_runs % 10 == 0:
            for toxicity_type in fps_best:
                convergence = fps_best[toxicity_type]<detoxify_max_fp[toxicity_type]
                print ('{}: FP={:.4f}% FN={:.4f}% @ th={:.6f} pass_quality={} target_error={:.4f}'.format(toxicity_type,fps_best[toxicity_type]*100.0, fns_best[toxicity_type]*100.0, ths_best[toxicity_type], convergence, target_errors_best[toxicity_type]))
            print ('opt-phase={}, weight_local_linear={}, target_error_best={:.5f}'.format(opt_phase,weight_local_linear, target_error_best))
    
            print ()

