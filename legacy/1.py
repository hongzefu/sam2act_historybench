import numpy as np
import pickle
import matplotlib
# 使用非交互式后端（适合服务器环境）
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

replay_dir = '/nfs/turbo/coe-chaijy-unreplicated/datasets/sam2act/buffer/replay_temporal/replay_train/close_jar'

# 加载前10个replay文件的front_rgb
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

success_count = 0
for i in range(10):
    replay_path = f"{replay_dir}/{i}.replay"
    try:
        with open(replay_path, "rb") as f:
            replay_data = pickle.load(f)
        
        # 获取front_rgb
        if 'front_rgb' in replay_data:
            front_rgb = replay_data['front_rgb']
            print(f"Replay {i}: front_rgb shape = {front_rgb.shape}, dtype = {front_rgb.dtype}, min = {front_rgb.min():.3f}, max = {front_rgb.max():.3f}")
            
            # front_rgb的格式可能是(3, H, W)或(H, W, 3)
            if front_rgb.shape[0] == 3:  # (3, H, W)格式
                # 转换为(H, W, 3)格式
                front_rgb = np.transpose(front_rgb, (1, 2, 0))
            
            # 处理不同的数据格式
            # 如果数据是归一化的(-1到1之间)，转换为0-255
            if front_rgb.min() >= -1.1 and front_rgb.max() <= 1.1:
                # 归一化到0-1，然后转换为0-255
                front_rgb = (front_rgb + 1.0) / 2.0 * 255.0
                front_rgb = np.clip(front_rgb, 0, 255).astype(np.uint8)
            # 如果数据是归一化的(0-1之间)，转换为0-255
            elif front_rgb.max() <= 1.1:
                front_rgb = (front_rgb * 255.0).astype(np.uint8)
            # 如果数据已经是0-255范围
            elif front_rgb.max() <= 255.1:
                front_rgb = np.clip(front_rgb, 0, 255).astype(np.uint8)
            else:
                # 其他情况，先归一化到0-255
                min_val = front_rgb.min()
                max_val = front_rgb.max()
                if max_val > min_val:
                    front_rgb = ((front_rgb - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
                else:
                    # 所有值相同，设为0
                    front_rgb = np.zeros_like(front_rgb, dtype=np.uint8)
            
            # 显示图像
            axes[i].imshow(front_rgb)
            axes[i].set_title(f'Replay {i}')
            axes[i].axis('off')
            success_count += 1
            print(f"成功加载和显示 Replay {i}")
        else:
            print(f"Replay {i}: 未找到'front_rgb'键，可用键: {list(replay_data.keys())[:10]}")
            axes[i].text(0.5, 0.5, f'No front_rgb\nin replay {i}', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].axis('off')
    except FileNotFoundError:
        print(f"Replay {i}: 文件不存在")
        axes[i].text(0.5, 0.5, f'File not\nfound {i}', 
                    ha='center', va='center', transform=axes[i].transAxes)
        axes[i].axis('off')
    except Exception as e:
        print(f"Replay {i}: 加载错误 - {e}")
        axes[i].text(0.5, 0.5, f'Error\nloading {i}', 
                    ha='center', va='center', transform=axes[i].transAxes)
        axes[i].axis('off')

plt.tight_layout()
plt.suptitle('前10个Replay的Front RGB图像', fontsize=16, y=1.02)

# 保存图像到文件
output_path = 'replay_front_rgb_visualization.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n成功可视化 {success_count}/10 个replay文件")
print(f"图像已保存到: {os.path.abspath(output_path)}")