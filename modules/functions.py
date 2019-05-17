import text_processing
import torch
import torch.nn.functional as F
from imp import reload
from text_processing import read_split_file_lyrics, BatchWrapper, generate_iterators_lyrics

#Taken and modified from lab/hw
def evaluate_DATN(src_model, tgt_model, data_iter, batch_size, source_flag = False):
    src_model.eval()
    tgt_model.eval()
    val_loss = 0
    total = 0
    total_acc = 0
    val_acc = 0
    for i in range(len(data_iter)):
        vectors, labels = next(data_iter.__iter__())
        
        if source_flag:
            src_attention, H_src, src_output = src_model(vectors)
            attention, output, HComb = tgt_model(vectors, src_attention, H_src)
        else:
            output = tgt_model(vectors)
        #_, predicted = torch.topk(output.data, k=2, dim=1)
        #predictions = torch.zeros(labels.size()).to("cuda")
        #predictions.scatter_(1, predicted, 1)
       
        val_acc += ((output[0].topk(2).indices == labels.topk(2).indices).sum()).item()
       
        total_acc += labels.size(0)*2
        val_loss += F.kl_div(output.log(), labels)
        total +=1
        
    return val_loss / total, val_acc / total_acc

#Taken and modified from lab/hw
def training_loop_DATN(batch_size, num_epochs, model, 
                       loss_, optim, training_iter, 
                       dev_iter, source_flag = False, verbose=True, learn_flag = True):
    
    epoch = 0
    total_batches = int(len(training_iter))
    dev_accuracies = []
    test_accuracies = []
    while epoch <= num_epochs:
        if verbose:
            print("Training...")
        for i in range(total_batches):
            model.train()
            vectors, labels = next(training_iter.__iter__())
            model.zero_grad()
            
            if source_flag:
            
                src_attention, H, output = model(vectors)
            
            else:
                
                output = model(vectors)
            if learn_flag:
                lossy = loss_(output.log(), labels)
                lossy.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optim.step()

        with torch.no_grad():
            model.eval()
            if verbose:
                print("Evaluating dev...")
            eval_loss, eval_acc = evaluate_DATN(model, dev_iter, batch_size, source_flag = source_flag)

            dev_accuracies.append(eval_acc)
        if verbose:
            print("Epoch %i; Loss %f; Dev loss %f, Dev acc: %f"  %(epoch, lossy.item(), eval_loss, eval_acc))
#         else:
#             if epoch % 5 == 0:
#                 print("Epoch %i; Loss %f; Dev loss %f, Dev acc: %f"  %(epoch, lossy.item(), eval_loss, eval_acc))
        epoch += 1    
    best_dev = max(dev_accuracies)
    
    if source_flag:
        
        return best_dev, src_attention, H
    
    else:
        
        return best_dev
    
    
    
def training_loop_DATN_full(batch_size, num_epochs, src_model, tgt_model, 
                       loss_, optim_src, optim_tgt, training_iter, lambda_,
                       dev_iter, test_iter, source_flag = False, verbose=True, src_learn_flag = True):
    
    epoch = 0
    total_batches = int(len(training_iter))
    dev_accuracies = []
    test_accuracies = []
    while epoch <= num_epochs:
        if verbose:
            print("Training...")
        for i in range(total_batches):
            src_model.train()
            tgt_model.train()
            
            vectors, labels = next(training_iter.__iter__())
       
            src_model.zero_grad()
            tgt_model.zero_grad()
           
            
            if source_flag:
            
                src_attention, H_src, src_output = src_model(vectors)
                
                tgt_attention, tgt_output, full_rep_out_HComb = tgt_model(vectors, src_attention, H_src)
            
            else:
                
                output = model(vectors)
                
            if src_learn_flag:
                
                lossy = loss_(src_output.log(), labels)
                lossy.backward()
                torch.nn.utils.clip_grad_norm_(src_model.parameters(), 5.0)
                optim_src.step()
            
            src_loss = -(src_output.log().sum())/(len(src_output.log()))
            
            kl_div_loss = loss_(tgt_output.log(), labels)
            
            cosine_sim_adj = lambda_*F.cosine_similarity(src_attention, tgt_attention, 0).squeeze().sum()
            
            lossy_tgt = src_loss + kl_div_loss + cosine_sim_adj
            lossy_tgt.backward()
            torch.nn.utils.clip_grad_norm(tgt_model.parameters(), 5.0)
            optim_tgt.step()

        with torch.no_grad():
            tgt_model.eval()
            if verbose:
                print("Evaluating dev...")
            eval_loss, eval_acc = evaluate_DATN(src_model = src_model, tgt_model = tgt_model, data_iter = dev_iter, batch_size = batch_size, source_flag = source_flag)
            print("Evaluating test..")
            test_loss, test_acc = evaluate_DATN(src_model = src_model, tgt_model = tgt_model, data_iter = test_iter, batch_size = batch_size, source_flag = source_flag)
            dev_accuracies.append(eval_acc)
            test_accuracies.append(test_acc)
        
        if verbose:
            print("Epoch %i; Loss %f; Dev loss %f; Dev acc: %f; Test acc: %f" %(epoch, lossy_tgt.item(), eval_loss, eval_acc, test_acc))
                                   
            
#         else:
#             if epoch % 5 == 0:
#                 print("Epoch %i; Loss %f; Dev loss %f, Dev acc: %f"  %(epoch, lossy.item(), eval_loss, eval_acc))
        epoch += 1    
    best_dev = max(dev_accuracies)
    
    if source_flag:
        
        return best_dev, src_attention, H_src
    
    else:
        
        return best_dev
    
    