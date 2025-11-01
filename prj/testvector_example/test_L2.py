import numpy as np, torch
import pandas as pd

#启动方式：python test_error.py

config = "5"

N, D = 64, 768
path_output= "/home/jicz/fpt_LLM/prj/testvector_example/bf16_vectors/hls_output_config_" + config + ".bin"
path_golden = "/home/jicz/fpt_LLM/prj/testvector_example/bf16_vectors/golden_out_config_"+ config + "_bf16.bin"


# 读作 uint16 并 reshape
output = np.fromfile(path_output, dtype=np.uint16).reshape(N, D)
golden = np.fromfile(path_golden, dtype=np.uint16).reshape(N, D)

if output.shape != golden.shape:
    raise ValueError(f"Shape mismatch: {output.shape} vs {golden.shape}")

# ✅ 先转换为 torch 支持的 dtype（int32），保持 bit pattern 不变
output_int16 = torch.from_numpy(output.view(np.int16))
golden_int16 = torch.from_numpy(golden.view(np.int16))


# ✅ 重新解释为 bfloat16
output_bf16 = output_int16.view(torch.bfloat16)
golden_bf16 = golden_int16.view(torch.bfloat16)


# ✅ 转换为 float32 便于计算
output_f32 = output_bf16.to(torch.float32).numpy()
golden_f32 = golden_bf16.to(torch.float32).numpy()

f32_max = np.finfo(np.float32).max
f32_min = -f32_max


# # 替换 nan -> 0, +inf -> f32_max, -inf -> f32_min
# output_f32 = np.nan_to_num(output_f32, nan=0.0, posinf=f32_max, neginf=f32_min)
# golden_f32 = np.nan_to_num(golden_f32, nan=0.0, posinf=f32_max, neginf=f32_min)

df_error = pd.DataFrame(output_f32)
# df_error.to_csv("/home/xushaohui/FPT/fpt_LLM/seedata/output_f32.csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名
df_error = pd.DataFrame(golden_f32)
# df_error.to_csv("/home/xushaohui/FPT/fpt_LLM/seedata/golden_f32.csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名
eps =1e-12
BIG_VAL = 1e9  # 惩罚用的大数，float32可以安全表达

#判断nan分布情况并输出结果，并将所有nan置零
nandif_out = np.empty((0, 2))
nandif_gold = np.empty((0, 2))

bad_count_row = np.zeros(N, dtype=np.int32)

for i in range(0,64):
    for j in range(0,768):
        out = output_f32[i,j]
        gold = golden_f32[i,j]
        if pd.isna(out) and pd.isna(gold):
            output_f32[i,j] = 0
            golden_f32[i,j] = 0
        elif pd.isna(out) and not pd.isna(gold):
            new_out = np.array([[i,j]])   # 注意这里要是二维形状 (1, 2)
            nandif_out = np.vstack((nandif_out, new_out))
            bad_count_row[i] += 1
            output_f32[i,j] = 0
            golden_f32[i,j] = 0 
        elif not pd.isna(out) and pd.isna(gold):
            new_gold = np.array([[i,j]])
            nandif_gold = np.vstack((nandif_gold, new_gold))
            bad_count_row[i] += 1
            output_f32[i,j] = 0
            golden_f32[i,j] = 0

df_nandif_out = pd.DataFrame(nandif_out)
df_nandif_gold = pd.DataFrame(nandif_gold)
df_nandif_out.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/df_nandif_out_config_"+ config +".csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名
df_nandif_gold.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/df_nandif_gold_config_"+ config +".csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名


#判断inf分布情况并输出结果，并将所有inf置零
infdif_out = np.empty((0, 2))
infdif_gold = np.empty((0, 2))
infdif_sign = np.empty((0, 2))

for i in range(0,64):
    for j in range(0,768):
        out = output_f32[i,j]
        gold = golden_f32[i,j]
        if np.isinf(out) and np.isinf(gold):
            if out != gold:
                bad_count_row[i] += 1
                new_sign= np.array([[i,j]]) 
                infdif_sign = np.vstack((infdif_sign, new_sign))
                output_f32[i, j] = 0
                golden_f32[i, j] = 0.0
            else:
                output_f32[i,j] = 0
                golden_f32[i,j] = 0
        elif np.isinf(out) and not np.isinf(gold):
            new_out = np.array([[i,j]])   # 注意这里要是二维形状 (1, 2)
            infdif_out = np.vstack((infdif_out, new_out))
            bad_count_row[i] += 1
            output_f32[i,j] = 0
            golden_f32[i,j] = 0 
        elif not np.isinf(out) and np.isinf(gold):
            new_gold = np.array([[i,j]])
            infdif_gold = np.vstack((infdif_gold, new_gold))
            bad_count_row[i] += 1
            output_f32[i,j] = 0
            golden_f32[i,j] = 0

df_infdif_out = pd.DataFrame(infdif_out)
df_infdif_gold = pd.DataFrame(infdif_gold)
df_infdif_sign = pd.DataFrame(infdif_sign)
df_infdif_out.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/df_infdif_out_config_"+ config +".csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名
df_infdif_gold.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/df_infdif_gold_config_"+ config +".csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名
df_infdif_sign.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/df_infdif_sign_config_"+ config +".csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名

#误差计算并计算精度得分
errors_per_row = []
errors_point_per_row = []
errors = 0
errors_point = 0

#计算评分
def calculate_point(error_true):
    ERROR_point= 0 
    if error_true <= 1e-3:
        ERROR_point = 1 
    elif error_true > 1e-3 and error_true <= 0.1:
        ERROR_point = (np.log(0.1) - np.log(error_true))/np.log(100)
    else:
        ERROR_point = 0
    return ERROR_point
   
#计算每行的真实误差和每行的评分
for f in range(N):
    # row_out  = output_f32[f]    # 形状: (D,)
    # row_gold = golden_f32[f]    # 形状: (D,)
    # rel_err_vec = np.abs(row_out - row_gold) / (np.abs(row_gold) + eps)   #计算向量中每个元素的相对误差 (D,)
    # base_scores = np.vectorize(calculate_point, otypes=[float])(rel_err_vec) # 依次计算每个元素的评分 (D,)
    # final_scores_vec = np.maximum(0.0, base_scores)
    # row_score = final_scores_vec.mean() - (bad_count_row[f] / D)
    # final_score = max(0.0, row_score)

    numerator = np.linalg.norm(output_f32[f] - golden_f32[f], ord=2)
    denominator = np.linalg.norm(golden_f32[f] , ord=2) + eps
    rel_err = numerator / denominator
    base_score = calculate_point(rel_err)
    penalty = bad_count_row[f] / D
    final_score = max(0.0, base_score - penalty)

    errors_per_row.append(rel_err)
    # errors_per_row.append(rel_err_vec.mean())
    errors_point_per_row.append(final_score)

# #计算总误差和总评分
# numerator_all   = np.linalg.norm(output_f32 - golden_f32, ord=2)
# denominator_all = np.linalg.norm(golden_f32 , ord=2) + eps
# errors = numerator_all / denominator_all

# errors_point = calculate_point(errors)

# t_int16 = torch.from_numpy(errors.view(np.int16))

# # ✅ 重新解释为 bfloat16
# t_bf16 = t_int16.view(torch.bfloat16)

# ✅ 转换为 float32 便于查看数值
# t_f32 = t_bf16.to(torch.float32).numpy()


df_error = pd.DataFrame(errors_per_row)
df_error_point = pd.DataFrame(errors_point_per_row)

# #输出总真实误差
# print("评分用真实误差：", errors)
# #输出总评分
# print("总评分：", errors_point)

# 将DataFrame输出为Excel文件
#df_error.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/errors_per_row_config_"+ config +".csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名
#df_error_point.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/errors_point_per_row_config_"+ config +".csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名

#输出使用进位算法的结果
df_error.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/errors_per_row_config_"+ config +"_round.csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名
df_error_point.to_csv("/home/jicz/fpt_LLM/prj/seedata/config_"+ config +"/errors_point_per_row_config_"+ config +"_round.csv", index=False, header=False)  # index=False 去掉行号，header=False 去掉列名

