"""
@author: Sotiris
"""
import os
import pandas as pd
import numpy as np
from collections import namedtuple
os.chdir(r'C:\Users\sotir\Documents\thesis')
from utils import LoggingConfig

def debate_segments_annotations(paths, valid_pids):
        
    subject_info_table = pd.read_csv(paths['subjects_info_path'], index_col='pid')
    Ratings = namedtuple('Ratings', ['values', 'len'])
    
    # for each participant:
    pid_info = {}
    for pid in valid_pids:
        pid_info[pid] = {}

        # get session info and timestamps
        subject_info = subject_info_table.loc[subject_info_table.index == pid]
        debate_start, debate_end = tuple(subject_info[['startTime', 'endTime']].values[0])
        debate_len = debate_end - debate_start
        
        # load self/partner/external annotation files
        ratings = {
            's': pd.read_csv(os.path.join(paths['self_ratings_dir'], f'P{pid}.self.csv')),
            'p': pd.read_csv(os.path.join(paths['partner_ratings_dir'], f'P{pid}.partner.csv')),
            'e': pd.read_csv(os.path.join(paths['external_ratings_dir'], f'P{pid}.external.csv'))
        }

        # save ratings information as (annotations, total duration of annotation in milliseconds)
        for tag in ['s', 'p', 'e']:
            ratings[tag] = Ratings(ratings[tag], int(ratings[tag].seconds.values[-1] * 1e3))

        # first, cut annotations longer than debate_len from their beginnings (1st step)
        for tag in ['s', 'p', 'e']:
            if ratings[tag].len >= debate_len:
                ratings[tag] = ratings[tag]._replace(values=ratings[tag].values[-int(debate_len // 5e3):].reset_index(drop=True))
                ratings[tag] = ratings[tag]._replace(len=int((ratings[tag].values.index[-1] + 1) * 5e3))

        # finally, cut annotations from their beginnings to match their lengths with each other
        min_len = min([ratings['s'].len, ratings['p'].len, ratings['e'].len])
        for tag in ['s', 'p', 'e']:
            if ratings[tag].len > min_len:
                ratings[tag] = ratings[tag]._replace(values=ratings[tag].values[int((ratings[tag].len - min_len) // 5e3):].reset_index(drop=True))
                ratings[tag] = ratings[tag]._replace(len=int((ratings[tag].values.index[-1] + 1) * 5e3))
        
        # store info in dictionary
        for tag in ['s', 'p', 'e']:
            pid_info[pid][tag] = ratings[tag][0]  
        
    return pid_info

def create_csv_files(paths, valid_pids, pid_info):
    
    # create separate dict for each perspective
    anno_self = {}
    anno_partner = {}
    anno_external = {}
    
    for pid in valid_pids:
        anno_self[pid] = {}
        anno_partner[pid] = {}
        anno_external[pid] = {}
        for tag in ['s', 'p', 'e']:
            if tag == 's':
                anno_self[pid] = pid_info[pid][tag]
            if tag == 'p':
                anno_partner[pid] = pid_info[pid][tag]
            if tag == 'e':
                anno_external[pid] = pid_info[pid][tag]
                
    # not-needed emotion labels and other columns
    col_drop = ['arousal', 'valence', 'boredom', 'confusion', 'delight', 
                'concentration', 'frustration', 'surprise', 'none_1', 'confrustion', 
                'contempt', 'dejection', 'disgust', 'eureka', 'pride', 'sorrow', 'none_2',
                'seconds']
                
    for perspective in [anno_self, anno_partner, anno_external]: 
        for pid in valid_pids:
            for col_label in col_drop:
                perspective[pid].drop(col_label, inplace=True, axis=1)
                
    # create PA, NA, unit, pid, occasions columns
    for perspective in [anno_self, anno_partner, anno_external]:
        for pid in valid_pids:
            pos_affect = []
            neg_affect = []
            unit = []
            occasions = []
            person = []
            anno = 0
            for index,row in perspective[pid].iterrows():
                anno += 1
                unit.append(1)
                occasions.append(anno)
                person.append(pid)
                pos_affect.append(np.mean([row['happy'], row['cheerful']], axis=0))
                neg_affect.append(np.mean([row['angry'], row['nervous'], row['sad']], axis=0))
            perspective[pid]['PID'] = person
            perspective[pid]['UNIT'] = unit
            perspective[pid]['OCCASION'] = occasions
            perspective[pid]['PA'] = pos_affect
            perspective[pid]['NA'] = neg_affect
    
    # concat DataFrames (dummy way)
    kemocon_self_df = pd.concat([anno_self[1], anno_self[2], anno_self[3], anno_self[4], anno_self[5],
                              anno_self[6], anno_self[7], anno_self[8], anno_self[9], anno_self[10],
                              anno_self[11], anno_self[12], anno_self[13], anno_self[14], anno_self[15],
                              anno_self[16], anno_self[17], anno_self[18], anno_self[19], anno_self[20],
                              anno_self[21], anno_self[22], anno_self[23], anno_self[24], anno_self[25],
                              anno_self[26], anno_self[27], anno_self[28], anno_self[29], anno_self[30],
                              anno_self[31], anno_self[32]
                              ])
    kemocon_partner_df = pd.concat([anno_partner[1], anno_partner[2], anno_partner[3], anno_partner[4], anno_partner[5],
                              anno_partner[6], anno_partner[7], anno_partner[8], anno_partner[9], anno_partner[10],
                              anno_partner[11], anno_partner[12], anno_partner[13], anno_partner[14], anno_partner[15],
                              anno_partner[16], anno_partner[17], anno_partner[18], anno_partner[19], anno_partner[20],
                              anno_partner[21], anno_partner[22], anno_partner[23], anno_partner[24], anno_partner[25],
                              anno_partner[26], anno_partner[27], anno_partner[28], anno_partner[29], anno_partner[30],
                              anno_partner[31], anno_partner[32]
                              ])
    kemocon_external_df = pd.concat([anno_external[1], anno_external[2], anno_external[3], anno_external[4], anno_external[5],
                              anno_external[6], anno_external[7], anno_external[8], anno_external[9], anno_external[10],
                              anno_external[11], anno_external[12], anno_external[13], anno_external[14], anno_external[15],
                              anno_external[16], anno_external[17], anno_external[18], anno_external[19], anno_external[20],
                              anno_external[21], anno_external[22], anno_external[23], anno_external[24], anno_external[25],
                              anno_external[26], anno_external[27], anno_external[28], anno_external[29], anno_external[30],
                              anno_external[31], anno_external[32]
                              ])    
    
    # set the columns at the right place 
    kemocon_self = kemocon_self_df[['PID', 'UNIT', 'OCCASION', 'PA', 'NA', 'cheerful', 'happy', 'angry', 'nervous','sad']]
    kemocon_partner = kemocon_partner_df[['PID', 'UNIT', 'OCCASION', 'PA', 'NA', 'cheerful', 'happy', 'angry', 'nervous','sad']]
    kemocon_external = kemocon_external_df[['PID', 'UNIT', 'OCCASION', 'PA', 'NA', 'cheerful', 'happy', 'angry', 'nervous','sad']]
    
    # save Kemocon.csv file for each perspective
    kemocon_self.to_csv(os.path.join(paths['out_dir'], 'self', 'kemocon_self.csv'), sep=';', header=True, index=False)
    kemocon_partner.to_csv(os.path.join(paths['out_dir'], 'partner', 'kemocon_partner.csv'), sep=';', header=True, index=False)
    kemocon_external.to_csv(os.path.join(paths['out_dir'], 'external', 'kemocon_external.csv'), sep=';', header=True, index=False)
    
    # save Kemocon_bound.csv file for each perspective
    lower_bound = 0
    upper_bound = 4
    bounds_dict = [{'LOWER': lower_bound, 'UPPER': upper_bound}]
    
    kemocon_self_bound = pd.DataFrame(bounds_dict)
    kemocon_partner_bound = pd.DataFrame(bounds_dict)
    kemocon_external_bound = pd.DataFrame(bounds_dict)
    
    kemocon_self_bound.to_csv(os.path.join(paths['out_dir'], 'self', 'kemocon_bound_self.csv'), sep=';', header=True, index=False)
    kemocon_partner_bound.to_csv(os.path.join(paths['out_dir'], 'partner', 'kemocon_bound_partner.csv'), sep=';', header=True, index=False)
    kemocon_external_bound.to_csv(os.path.join(paths['out_dir'], 'external', 'kemocon_bound_external.csv'), sep=';', header=True, index=False)

    # save Kemocon_val.csv file for each perspective
    lower_val = -1 
    upper_val = 1 
    val_dict = [{'happy': upper_val, 'cheerful': upper_val, 'angry': lower_val, 'nervous': lower_val, 'sad': lower_val}]
    
    kemocon_self_val = pd.DataFrame(val_dict)
    kemocon_partner_val = pd.DataFrame(val_dict)
    kemocon_external_val = pd.DataFrame(val_dict)
    
    kemocon_self_val.to_csv(os.path.join(paths['out_dir'], 'self', 'kemocon_val_self.csv'), sep=';', header=True, index=False)
    kemocon_partner_val.to_csv(os.path.join(paths['out_dir'], 'partner', 'kemocon_val_partner.csv'), sep=';', header=True, index=False)
    kemocon_external_val.to_csv(os.path.join(paths['out_dir'], 'external', 'kemocon_val_external.csv'), sep=';', header=True, index=False)
        
    # save Kemocon_cov.csv file for each perspective (is this necessary?)
    
if __name__ == '__main__':
    logger = LoggingConfig('info', handler_type='stream').get_logger()
    PATHS = {
            'subjects_info_path':(r'C:\Users\sotir\Documents\thesis\dataset\metadata\subjects.csv'),
            'out_dir': (r'C:\Users\sotir\Documents\thesis\affect_dynamics'),
            'self_ratings_dir': (r'C:\Users\sotir\Documents\thesis\dataset\emotion_annotations\self_annotations'),
            'partner_ratings_dir': (r'C:\Users\sotir\Documents\thesis\dataset\emotion_annotations\partner_annotations'),
            'external_ratings_dir': (r'C:\Users\sotir\Documents\thesis\dataset\emotion_annotations\aggregated_external_annotations')
            }

    VALIDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
    
    # preprocess annotations and return dict
    logger.info('Preprocess to get emotions corresponding to debate segments')
    pid_info = debate_segments_annotations(PATHS, VALIDS)
    logger.info('Annotations preprocessing complete')
    
    # create csv files for all perspectives
    logger.info('Creating and saving csv files')
    create_csv_files(PATHS, VALIDS, pid_info)
    
    
    

