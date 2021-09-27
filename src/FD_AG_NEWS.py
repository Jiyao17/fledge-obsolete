
import torch
from torchtext.datasets import AG_NEWS
from torch.utils.data import DataLoader
from torch import nn
import time

from multiprocessing.context import Process
from multiprocessing import Process, Queue, set_start_method

from utils.model import AG_NEWS_TEXT
from utils.funcs import get_argument_parser, check_device, get_partitioned_datasets, get_test_dataset

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)


class ClientAGNEWS:
    def __init__(self, dataset, testset, l_epoch, batch_size, lr) -> None:
        self.test_dataloader = DataLoader(testset, batch_size=self.batch_size,
                                    shuffle=True, collate_fn=self.collate_batch)
        self.batch_size = batch_size
        self.lr = lr
        self.l_epoch = l_epoch

        from torchtext.data.utils import get_tokenizer
        from torchtext.vocab import build_vocab_from_iterator

        self.tokenizer = get_tokenizer('basic_english')
        self.train_iter = AG_NEWS(split='train')
        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.train_iter), specials=["<unk>"])
        self.vocab.set_default_index(self.vocab["<unk>"])
        
        self.text_pipeline = lambda x: self.vocab(self.tokenizer(x))
        self.label_pipeline = lambda x: int(x) - 1


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # self.dataloader = None
        # train_iter = AG_NEWS(split='train')
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=self.collate_batch)

        train_iter = AG_NEWS(split='train')
        self.num_class = len(set([label for (label, text) in train_iter]))
        self.vocab_size = len(self.vocab)
        self.emsize = 64
        self.model = AG_NEWS_TEXT(self.vocab_size, self.emsize, self.num_class).to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.1)


    def yield_tokens(self, data_iter):
        for _, text in data_iter:
            yield self.tokenizer(text)

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)

    def train(self):
        self.model.train()
        total_acc, total_count = 0, 0

        for idx, (label, text, offsets) in enumerate(self.dataloader):
            self.optimizer.zero_grad()
            predicted_label = self.model(text, offsets)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
        
        return total_acc/total_count
            # if idx % log_interval == 0 and idx > 0:
            #     elapsed = time.time() - start_time
            #     print('| epoch {:3d} | {:5d}/{:5d} batches '
            #         '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
            #                                     total_acc/total_count))
            #     total_acc, total_count = 0, 0
            #     start_time = time.time()

    def evaluate(self, dataloader):
        self.model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = self.model(text, offsets)
                loss = self.criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count

    def run(self):
        # import time

        # from torch.utils.data.dataset import random_split
        # from torchtext.data.functional import to_map_style_dataset

        # train_iter, test_iter = AG_NEWS()
        # train_dataset = to_map_style_dataset(train_iter)
        # test_dataset = to_map_style_dataset(test_iter)

        # train_dataloader = DataLoader(self.dataloader, batch_size=self.batch_size,
        #                             shuffle=True, collate_fn=self.model.collate_batch)
        # test_dataloader = DataLoader(self.testset, batch_size=self.batch_size,
        #                             shuffle=True, collate_fn=self.model.collate_batch)

        for epoch in range(self.l_epoch):
            # epoch_start_time = time.time()
            self.train()
            # accu_val = evaluate(valid_dataloader)
            # if total_accu is not None and total_accu > accu_val:
            #   scheduler.step()
            # else:
            #    total_accu = accu_val
            # print('-' * 59)
            # print('| end of epoch {:3d} | time: {:5.2f}s | '
            #     'valid accuracy {:8.3f} '.format(epoch,
            #                                     time.time() - epoch_start_time,
            #                                     accu_val))
            # print('| end of epoch {:3d} | time: {:5.2f}s | '.format(epoch,
            #                                     time.time() - epoch_start_time,
            #                                     ))
            # print('-' * 59)

        # print('Checking the results of test dataset.')
        # accu_test = self.evaluate(test_dataloader)
        # print('test accuracy {:8.3f}'.format(accu_test))

def run_sim(que: Queue, progress_file: str, task, g_epoch_num, client_num, l_data_num, l_epoch_num, l_batch_size, l_lr, data_path, device, verbosity):
    # partition data
    datasets = get_partitioned_datasets(task, client_num, l_data_num, l_batch_size, data_path)
    test_dataset = get_test_dataset(task, data_path)

    ClientAGNEWS(datasets[1])

    if verbosity >= 1:
        pf = open(progress_file, "a")
        print(f"Global accuracy:{g_accuracy*100:.9f}%")
        pf.write(f"Epoch {i}: {g_accuracy*100:.2f}%\n")
        # if i % 10 == 9:
        pf.flush()
        pf.close()
        # print(f"Local accuracy after training: {[acc for acc in l_accuracy]}")
    
    if i % 10 == 9:
        result.append(g_accuracy)

    que.put(result)



if __name__ == "__main__":

    ap = get_argument_parser()
    args = ap.parse_args()

    TASK: str = args.task # limited: FashionMNIST/SpeechCommand/
    # global parameters
    G_EPOCH_NUM: int = args.g_epoch_num
    # local parameters
    CLIENT_NUM: int = args.client_num
    L_DATA_NUM: int = args.l_data_num
    L_EPOCH_NUM: int = args.l_epoch_num
    L_BATCH_SIZE: int = args.l_batch_size
    L_LR: float = args.l_lr
    # shared settings
    DATA_PATH: str = args.datapath
    DEVICE: str = torch.device(args.device)
    RESULT_FILE: str = args.result_file
    VERBOSITY: int = args.verbosity
    RUN_NUM: int = args.run_num
    PROGRESS_FILE: str = args.progress_file

    if VERBOSITY >= 2:
        print("Input args: %s %d %d %d %d %d %f %s %s %s" %
            (TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, RESULT_FILE)
            )

    # input check
    SUPPORTED_TASKS = ["FashionMNIST", "SpeechCommand", "AG_NEWS"]
    if TASK not in SUPPORTED_TASKS:
        raise "Task not supported!"
    if check_device(DEVICE) == False:
        raise "CUDA required by input but not equipped!"

    # run_sim(Queue(), TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, RESULT_FILE, VERBOSITY)
    # exit()

    set_start_method("spawn")
    que = Queue()
    procs: 'list[Process]' = []

    for i in range(RUN_NUM):
        proc = Process(
                target=run_sim,
                args=(que, PROGRESS_FILE, TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR, DATA_PATH, DEVICE, VERBOSITY)
            )
        proc.start()
        procs.append(proc)

    for proc in procs:
        proc.join()

    with open(RESULT_FILE, "a") as f:
        args = "{:12} {:11} {:10} {:10} {:11} {:12} {:4}".format(
            TASK, G_EPOCH_NUM, CLIENT_NUM, L_DATA_NUM, L_EPOCH_NUM, L_BATCH_SIZE, L_LR
            )
        f.write(
            "TASK          G_EPOCH_NUM CLIENT_NUM L_DATA_NUM L_EPOCH_NUM L_BATCH_SIZE L_LR\n" +
            args + "\n"
            )

        for i in range(RUN_NUM):
            result = que.get(block=True)
            print(result)
            [f.write(f"{num*100:.2f}% ") for num in result]
            f.write("\n")
            f.flush()
        
        f.write("\n")