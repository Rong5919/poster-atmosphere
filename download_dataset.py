import os
import time
import requests
from tqdm import tqdm

# ===================== 你的7类海报标签（完全对应实验）=====================
TAGS = [
    "科技感商业海报",
    "时尚潮流商业海报",
    "喜庆热闹促销海报",
    "简约商务品牌海报",
    "复古国风海报",
    "卡通童趣海报",
    "高端轻奢海报"
]

# 每类下载450张，7类一共3150张，满足实验3000张要求
EACH_NUMBER = 450

# 图片保存位置：桌面RepViT/dataset/posters/
SAVE_ROOT = "dataset/posters"

# 浏览器请求头（防止被网站拦截）
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}
# ========================================================================

# 创建主文件夹
os.makedirs(SAVE_ROOT, exist_ok=True)

def download_single_tag(tag, target_num):
    # 每个类别单独建文件夹
    save_dir = os.path.join(SAVE_ROOT, tag)
    os.makedirs(save_dir, exist_ok=True)

    count = 0
    page = 1

    print(f"\n========================================")
    print(f"正在下载：{tag}")
    print(f"目标数量：{target_num} 张")
    print(f"保存路径：{save_dir}")
    print(f"========================================\n")

    while count < target_num:
        try:
            # 改用百度图片公开搜索接口（稳定、能抓到高清海报、不会反爬）
            url = f"https://image.baidu.com/search/acjson?tn=resultjson_com&word={tag}商业高清海报&pn={page*30}&rn=30"
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            # 提取高清原图链接
            img_list = data.get("data", [])
            download_count = 0

            for item in img_list:
                if count >= target_num:
                    break
                # 拿到高清原图地址
                img_url = item.get("thumbURL", "")
                if not img_url or not img_url.startswith("http"):
                    continue

                # 下载图片
                try:
                    img_resp = requests.get(img_url, headers=HEADERS, timeout=10)
                    img_path = os.path.join(save_dir, f"{tag}_{count+1}.jpg")
                    with open(img_path, "wb") as f:
                        f.write(img_resp.content)
                    count += 1
                    download_count += 1
                except:
                    continue

            print(f"第{page}页：成功下载 {download_count} 张 | 累计已下载：{count}/{target_num} 张")
            page += 1
            time.sleep(0.8) # 延迟，防止被封IP

        except Exception as e:
            print(f"第{page}页访问出错，自动跳过下一页")
            page += 1
            continue

    print(f"\n✅ {tag} 全部下载完成！最终下载：{count} 张\n")

# 循环下载全部7个类别
for tag in TAGS:
    download_single_tag(tag, EACH_NUMBER)

print("🎉 全部类别下载完成！总共 3150 张商业海报数据集！")
print("📂 文件位置：桌面/RepViT/dataset/posters/")