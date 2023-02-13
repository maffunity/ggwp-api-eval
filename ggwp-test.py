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

api_key = '3_Hb2c23mrmtsJSIvbODo2aditTDh7Wt5Linsdmgq-rY1aKvrbS5sdaru9iorralt9OotavWoeSq2qaqgHqAgnt1vG0'
base_url = 'https://api.ggwp.com/chat/v2/message'
voice_actor_dataset_path='/home/ext_marc_ferras_unity3d_com/project/corpora/unity-oto-ml-prd-us-central1-innodata/batch_1/innodata_results/'

# original config
# config = {
#     "violence": "off",
#     "sexual_content": "low",
#     "verbal_abuse": "medium",
#     "identity_hate": "medium",
#     "profanity": "high",
#     "link_sharing": "high",
#     "drugs": "off",
#     "spam": "on",
#     "self_harm": "on",
#     "replace": "off",
# }

# after tuning, sexual_content=medium obtains best result against our GT
# that's the only change to the configuration
# config = {
#     "violence": "off",
#     "sexual_content": "high",
#     "verbal_abuse": "medium",
#     "identity_hate": "medium",
#     "profanity": "high",
#     "link_sharing": "high",
#     "drugs": "off",
#     "spam": "on",
#     "self_harm": "on",
#     "replace": "off",
# }
config = {
    "violence": "off",
    "sexual_content": "medium",
    "verbal_abuse": "medium",
    "identity_hate": "medium",
    "profanity": "medium",
    "link_sharing": "high",
    "drugs": "off",
    "spam": "on",
    "self_harm": "on",
    "replace": "off",
}

def get_short_config_name(config):

    violence = '1' if config['violence'].lower()=='on' else '0'
    drugs = '1' if config['drugs'].lower()=='on' else '0'
    self_harm = '1' if config['self_harm'].lower()=='on' else '0'

    if config['sexual_content'].lower() == 'low':
        sexual_content = 'L'
    elif config['sexual_content'].lower() == 'medium':
        sexual_content = 'M'
    elif config['sexual_content'].lower() == 'high':
        sexual_content = 'H'

    if config['verbal_abuse'].lower() == 'low':
        verbal_abuse = 'L'
    elif config['verbal_abuse'].lower() == 'medium':
        verbal_abuse = 'M'
    elif config['verbal_abuse'].lower() == 'high':
        verbal_abuse = 'H'

    if config['profanity'].lower() == 'low':
        profanity = 'L'
    elif config['profanity'].lower() == 'medium':
        profanity = 'M'
    elif config['profanity'].lower() == 'high':
        profanity = 'H'

    if config['identity_hate'].lower() == 'low':
        identity_hate = 'L'
    elif config['identity_hate'].lower() == 'medium':
        identity_hate = 'M'
    elif config['identity_hate'].lower() == 'high':
        identity_hate = 'H'

    config_short_name = 'v{}sc{}va{}ih{}p{}d{}sh{}'.format(
        violence,
        sexual_content,
        verbal_abuse,
        identity_hate,
        profanity,
        drugs,
        self_harm,
    )
    
    return config_short_name 


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

def ggwp_request (text, config):
    config_base64 = base64.b64encode(bytes(json.dumps(config, indent = 2), 'ascii')).decode()
    header = {
        'x-api-config': config_base64,
        'x-api-key': api_key,
        'Content-Type': 'application/json',
        }
    data = {
        "session_id": "test_session",
        "message": text,
        "user_id": "test_user",
        "username": "test_username",
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    response = post(base_url, headers=header, json=data)
    if response.status_code == 200:
        result = json.loads(response.text)
        input_text = result['message_details']['original_message']
        filtered_text = result['message_details']['filtered_message']
        return (result['message_details'])
    else:
        return None

def shuffle_sentence(text):

    texts = text
    texts = texts.split()
    random.shuffle(texts)
    return ' '.join(texts)


def ggwp_normalize_toxicity(result, human_label=False):
    """
    Return whether toxic and a list of types of toxicity
    """
    if result is None:
        return None, None

    toxicity = []
    if result['violence']:
        if human_label:
            toxicity.append('Severe Toxicity')
        else:
            toxicity.append('violence')
    if result['verbal_abuse']:
        if human_label:
            toxicity.append('Insult')
        else:        
            toxicity.append('verbal_abuse')
    if result['profanity']:
        if human_label:
            toxicity.append('Insult')
        else:
            toxicity.append('profanity')
    if result['sexual_content']:
        if human_label:
            toxicity.append('Obscene')
        else:
            toxicity.append('sexual_content')
    if result['identity_hate']:
        if human_label:
            toxicity.append('Identity comments')
        else:
            toxicity.append('identity_hate')
    if result['drugs']:
        if not human_label:
            toxicity.append('drugs')
    if result['self_harm']:
        if human_label:
            toxicity.append('Threat')
        else:
            toxicity.append('self_harm')
    if result['filtered_message']:
        filtered_message = result['filtered_message']
    else:
        filtered_message = None

    toxicity = [ t.replace(' ','_') for t in toxicity ]
    return len(toxicity)>0, sorted(toxicity), filtered_message


def detoxify_normalize_toxicity(result, threshold=0.5):
    """
    Return whether toxic and a list of types of toxicity
    """
    toxicity = []
    if 'obscene' in result and result['obscene']>threshold:
        toxicity.append ('obscene')
    if 'identity_attack' in result and result['identity_attack']>threshold:
        toxicity.append ('identity_attack')
    if 'insult' in result and result['insult']>threshold:
        toxicity.append ('insult')
    if 'threat' in result and result['threat']>threshold:
        toxicity.append ('threat')
    toxicity = [ t.replace(' ','_') for t in toxicity ]
    return len(toxicity)>0, sorted(toxicity)


def ggwp_toxicity_to_human_label(predict_toxicity_type):
    """
    Return whether toxic and a list of types of toxicity
    """
    toxicity = []
    if 'violence' in predict_toxicity_type:
        toxicity.append('Severe Toxicity')
    if 'verbal_abuse' in predict_toxicity_type:
        toxicity.append('Insult')
    if 'profanity' in predict_toxicity_type:
        toxicity.append('Insult')
    if 'sexual_content' in predict_toxicity_type:
        toxicity.append('Obscene')
    if 'identity_hate' in predict_toxicity_type:
        toxicity.append('Identity comments')
    if 'drugs' in predict_toxicity_type:
        toxicity.append('drugs')
    if 'self_harm' in predict_toxicity_type:
        toxicity.append('Threat')
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

    def add(self, text, toxic, toxicity_types, filtered_message=None):
        if filtered_message is None: 
            self.cache[text] = (toxic, toxicity_types)
        else:
            self.cache[text] = (toxic, toxicity_types, filtered_message)
        self.n_entries = len(self.cache)
        if self.fp is not None:
            if filtered_message is None: 
                self.fp.write('{}|{}|{}\n'.format(text, toxic, ','.join(toxicity_types)))
            else:
                self.fp.write('{}|{}|{}|{}\n'.format(text, toxic, ','.join(toxicity_types), filtered_message))
            self.fp.flush()

    def get(self, text):
        if text in self.cache:
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
                if len(parts)>=3:
                    text = parts[0]
                    toxic = True if parts[1].lower() in ("yes", "true", "t", "1") else False
                    toxicity_type = parts[2].split(',')
                    if len(parts)==4:
                        filtered_message = parts[3]
                    else:
                        filtered_message = None
                    self.cache[text] = (toxic, toxicity_type, filtered_message)


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
    system='ggwp'
    # system='detoxify'
    if system=='detoxify':
        from detoxify import Detoxify
        detoxify_model = Detoxify('original')
    else:
        detoxify_model = None

    exp_name = '{}_eval_{}_{}'.format(
                system,
                norm_config_short_name,
                get_short_config_name(config) )

    if per_toxicity_type:
        if human_label:
            toxicity_types = ['all','Identity_comments', 'Obscene', 'Insult', 'Threat']
        else:
            toxicity_types = ['all','verbal_abuse', 'profanity', 'sexual_content', 'identity_hate', 'self_harm']
    else:
        toxicity_types = ['all']

    for toxicity_type in toxicity_types:
        if len(toxicity_type)!='all':
            print ('\nevaluating toxicity type {} on system {}'.format(toxicity_type, system.upper()))
        else:
            print ('\nevaluating any toxicity on system {}'.format(system.upper()))

        toxicity_type_str = '_'+toxicity_type if toxicity_type!='all' else ''
        human_label_str = '_hl1' if human_label else ''
        # eval
        fp_correction=0.9
        tp = 0 ; fp = 0 ; fn = 0 ; tn = 0 ; n_egs = 0
        # print ('opening cache file {}'.format(os.path.join('stats',exp_name+'.cache')))
        # sys.exit()
        cache = Cache(os.path.join('stats',exp_name+'.cache'))
        stats_filename = os.path.join('stats',exp_name+'{}{}.stats'.format(human_label_str, toxicity_type_str))
        fp_stats = open(stats_filename,'wt')
        print ('check stats file {}'.format(stats_filename))
        for key, metadata in voice_actor_dataset.items():
            text = metadata['text']
            gt_toxic, gt_toxicity_type = voice_actor_normalize_toxicity(metadata['toxicity'], human_label=human_label)
            if toxicity_type!='all':
                gt_toxic = toxicity_type in gt_toxicity_type

            if cache.get(text) is None:
                if system =='ggwp':
                    result = ggwp_request (text, config)
                    if result is None:
                        print ('GGWP query failed!')
                        sys.exit()
                    # cache using raw GGWP labels, convert to human labels later
                    predict_toxic, predict_toxicity_type, filtered_message = ggwp_normalize_toxicity(result, human_label=False)
                    cache.add(text, predict_toxic, predict_toxicity_type, filtered_message)
                    time.sleep(0.5* random.random())
                else:
                    result = detoxify_model.predict(text)
                    predict_toxic, predict_toxicity_type = detoxify_normalize_toxicity(result)
                    cache.add(text, predict_toxic, predict_toxicity_type)
            else:
                predict_toxic, predict_toxicity_type, filtered_message = cache.get(text)
            
            # use human labels or GGWP label
            if human_label:
                if system=='ggwp':
                    predict_toxic, predict_toxicity_type = ggwp_toxicity_to_human_label (predict_toxicity_type)
                elif system =='detoxify':
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
                    fp_stats.write('\'{}\': FALSE-NEGATIVE, Human:{}({}), {}:{}({})\n'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(gt_toxicity_type), predict_toxic))
                    if not per_toxicity_type:
                        print('\'{}\': FALSE_NEGATIVE, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(gt_toxicity_type), predict_toxic))
                else:
                    fp_stats.write('\'{}\': FALSE-POSITIVE ERROR, Human:{}({}), {}:{}({})\n'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
                    if not per_toxicity_type:
                        print('\'{}\': FALSE-POSITIVE ERROR, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
            else:
                if gt_toxic and predict_toxic:
                    fp_stats.write('\'{}\': TRUE-POSITIVE, Human:{}({}), {}:{}({})\n'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
                    if not per_toxicity_type:
                        print('\'{}\': TRUE-POSITIVE, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
                else:
                    fp_stats.write('\'{}\': TRUE-NEGATIVE, Human:{}({}), {}:{}({})\n'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))
                    if not per_toxicity_type:
                        print('\'{}\': TRUE-NEGATIVE, Human:{}({}), {}:{}({})'.format(text, ','.join(gt_toxicity_type), gt_toxic, system.upper(), ','.join(predict_toxicity_type), predict_toxic))

            if max_egs is not None and n_egs>=max_egs:
                break

        print ('Toxicity type: {}'.format(toxicity_type))
        print ('Counts: tp={}, tn={}, fp={}, fn={}, total={}'.format(tp, tn, fp, fn, tp+tn+fp+fn))
        fp_rate = fp/n_egs
        fp_rate_corrected = fp*fp_correction/n_egs
        fn_rate = fn/n_egs
        if (tp+fp)>0:
            prec = tp/(tp+fp)
            prec_corrected = tp/(tp+fp*fp_correction)
        else:
            prec = math.inf
            prec_corrected = math.inf
        if (tp+fn)>0:
            rec = tp/(tp+fn)
        else:
            rec = math.inf
        F1 = 2 * (prec*rec) / (prec+rec)
        F1_corrected = 2 * (prec_corrected*rec) / (prec_corrected+rec)
        print ('FP-rate: {:.1f}%, FN-rate: {:.1f}%, Precision: {:.1f}%, Recall: {:.1f}% F1-score={:.3f}'.format( fp_rate*100.0, fn_rate*100.0, prec*100.0, rec*100.0, F1))
        print ('FP-rate: {:.1f}%, FN-rate: {:.1f}%, Precision: {:.1f}%, Recall: {:.1f}% F1-score={:.3f} FA-Correction={:.1f}'.format( fp_rate_corrected*100.0, fn_rate*100.0, prec_corrected*100.0, rec*100.0, F1_corrected, fp_correction))

        fp_stats.write('Toxicity type: {}\n'.format(toxicity_type))
        fp_stats.write('Counts: tp={}, tn={}, fp={}, fn={}, total={}\n'.format(tp, tn, fp, fn, tp+tn+fp+fn))
        fp_stats.write('FP-rate: {:.1f}%, FN-rate: {:.1f}%, Precision: {:.1f}%, Recall: {:.1f}% F1-score={:.3f}\n'.format(fp_rate*100.0, fn_rate*100.0, prec*100.0, rec*100.0, F1))
        fp_stats.write('FP-rate: {:.1f}%, FN-rate: {:.1f}%, Precision: {:.1f}%, Recall: {:.1f}% F1-score={:.3f} FA-Correction={:.1f}\n'.format( fp_rate_corrected*100.0, fn_rate*100.0, prec_corrected*100.0, rec*100.0, F1_corrected, fp_correction))
        fp_stats.close()
