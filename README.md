# 说明

## 视频地址

- **哔哩哔哩**
  https://www.bilibili.com/video/BV1A2tnzSEsv
- **Youtube**
  https://youtu.be/q_zh0-NcW9M

## 模型列表

- **artifact_en**  
  - 说明：使用莎士比亚作品训练的模型  
  - 文本文件：`input.txt`

- **artifact_cn**  
  - 说明：使用《西游记》训练的模型  
  - 文本文件：`西游记.txt`

- **artifact_4in1**  
  - 说明：使用四大名著合集训练的模型  
  - 文本文件：`4in1.txt`

## 使用方法

1. 将任意模型文件夹复制一份。
2. 重命名为 `artifact`。
3. 修改代码中的以下语句，将文件名替换为对应模型的文本文件名：

   ```rust
   let text = include_str!("4in1.txt");
   
## 另外
note文件中是我视频中画的图的原始文件，用Microsoft OneNote可以打开。另外note.mht文件是单一网页，没有安装OneNote的朋友可以直接看这个。



