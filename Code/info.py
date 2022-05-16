import os
import json


class Info:
    
    def __init__(self, args):
               
        self.DEVICE_CPU = 'cpu'
        self.DEVICE_GPU = 'cuda:0'
                
        self.MARKER_ENTITY = '*'
        self.EXTREME_SMALL_POSI = 1e-10
        self.EXTREME_SMALL_NEGA = -1e10
        
        self.TASK_RE = 'RE'
        self.TASK_CR = 'CR'
        self.TASK_ET = 'ET'
        self.TASK_PER = 'PER'
        self.TASK_FER = 'FER'
                
        self.STAGE_PREPARE = 'Prepare'
        self.STAGE_MAIN = 'Main'
        
        self.INFER_ROUND_FIRST = 'First'
        self.INFER_ROUND_FER = 'FER'
        self.INFER_ROUND_MASK = 'Mask'
        self.INFER_ROUND_DOC = 'Doc'
        self.INFER_ROUND_BLEND = 'Blend'
        
        self.DATASET = args.dataset        
        self.DIR_CURR = os.getcwd()
        self.DIR_DATA = os.path.join(self.DIR_CURR, '../Data')
        self.DIR_DATASET = os.path.join(self.DIR_DATA, self.DATASET)
        self.DIR_DATASET_ORI = os.path.join(self.DIR_DATASET, 'Original')
        self.DIR_DATASET_PRO = os.path.join(self.DIR_DATASET, 'Processed')
        self.DIR_DATASET_STD = os.path.join(self.DIR_DATASET, 'Stdout')

        self.FILE_STDOUT = os.path.join(self.DIR_DATASET_STD, f'stdout_{args.stage}.txt')
        self.FILE_MODEL = os.path.join(self.DIR_DATASET_PRO, f'model.pt')
                            
        if self.DATASET == 'DocRED':
            self.load_DocRED()
                    
    
    def load_DocRED(self):
        
        self.MODE_TRAIN = 'train_annotated'
        self.MODE_DEV = 'dev'
        self.MODE_TEST = 'test'
        
        self.FILE_CORPUS_TRAIN = os.path.join(self.DIR_DATASET_ORI, f'{self.MODE_TRAIN}.json')
        self.FILE_CORPUS_DEV = os.path.join(self.DIR_DATASET_ORI, f'{self.MODE_DEV}.json')
        self.FILE_CORPUS_TEST = os.path.join(self.DIR_DATASET_ORI, f'{self.MODE_TEST}.json')
        self.FILE_CORPUSES = {self.MODE_TRAIN:self.FILE_CORPUS_TRAIN, self.MODE_DEV:self.FILE_CORPUS_DEV, self.MODE_TEST:self.FILE_CORPUS_TEST}
        
        self.FILE_INPUT_TRAIN = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_TRAIN}_inputs.pkl')
        self.FILE_INPUT_DEV = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_DEV}_inputs.pkl')
        self.FILE_INPUT_TEST = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_TEST}_inputs.pkl')
        self.FILE_INPUTS = {self.MODE_TRAIN:self.FILE_INPUT_TRAIN, self.MODE_DEV:self.FILE_INPUT_DEV, self.MODE_TEST:self.FILE_INPUT_TEST}
        
        self.FILE_RESULT_TRAIN = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_TRAIN}_results')
        self.FILE_RESULT_DEV = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_DEV}_results')
        self.FILE_RESULT_TEST = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_TEST}_results')
        self.FILE_RESULTS = {self.MODE_TRAIN:self.FILE_RESULT_TRAIN, self.MODE_DEV:self.FILE_RESULT_DEV, self.MODE_TEST:self.FILE_RESULT_TEST}
        
        self.FILE_TRUTH_TRAIN = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_TRAIN}_truth.pkl')
        self.FILE_TRUTH_DEV = os.path.join(self.DIR_DATASET_PRO, f'{self.MODE_DEV}_truth.pkl')
        self.FILE_TRUTHS = {self.MODE_TRAIN:self.FILE_TRUTH_TRAIN, self.MODE_DEV:self.FILE_TRUTH_DEV}
        
        self.FILE_DEV_IN_TRAIN = os.path.join(self.DIR_DATASET_PRO, 'dev_in_train.pkl')
        self.FILE_DEV_INS = {self.MODE_TRAIN:self.FILE_DEV_IN_TRAIN}
        
        self.DATA_NER2ID = json.load(open(os.path.join(self.DIR_DATASET_ORI, 'ner2id.json'), 'r'))
        self.DATA_REL2ID = json.load(open(os.path.join(self.DIR_DATASET_ORI, 'rel2id.json'), 'r'))
        self.DATA_ID2NER = {v:k for k,v in self.DATA_NER2ID.items()}
        self.DATA_ID2REL = {v:k for k,v in self.DATA_REL2ID.items()}
        
        self.DATA_NER2WORD = {'BLANK': 'blank', 'ORG': 'organization', 'LOC': 'location', 'TIME': 'time', 'PER': 'person', 'MISC': 'miscellanea', 'NUM': 'number'}
        self.DATA_REL2WORD = json.load(open(os.path.join(self.DIR_DATASET_ORI, 'rel_info.json'), 'r'))
        self.DATA_NER2VEC = os.path.join(self.DIR_DATASET_PRO, 'ner2vec.npy')
        self.DATA_REL2VEC = os.path.join(self.DIR_DATASET_PRO, 'rel2vec.npy')
        
        self.ID_REL_THRE = self.DATA_REL2ID['Na']
        self.NUM_NER = len(self.DATA_NER2ID)
        self.NUM_REL = len(self.DATA_REL2ID)
