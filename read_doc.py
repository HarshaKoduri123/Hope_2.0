import os
import pdfplumber
import fitz
from typing import List, Dict, Any


def read_documents(data_dir: str, image_output_dir: str = "extracted_images") -> List[Dict[str, Any]]:


    documents = []

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    os.makedirs(image_output_dir, exist_ok=True)

    for filename in sorted(os.listdir(data_dir)):
        if not filename.lower().endswith(".pdf"):
            continue

        file_path = os.path.join(data_dir, filename)
        doc_id = os.path.splitext(filename)[0]

        text_pages = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_pages.append(page_text)

        content = "\n".join(text_pages).strip()
        image_metadata = []
        pdf_doc = fitz.open(file_path)

        for page_index in range(len(pdf_doc)):
            page = pdf_doc[page_index]
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_doc.extract_image(xref)

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_filename = f"{doc_id}_page{page_index+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(image_output_dir, image_filename)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                image_metadata.append({
                    "page": page_index + 1,
                    "image_path": image_path,
                    "image_ext": image_ext
                })

        pdf_doc.close()

        if content:
            documents.append({
                "id": doc_id,
                "content": content,
                "source": filename,
                "num_pages": len(text_pages),
                "images": image_metadata
            })

    if not documents:
        raise ValueError(f"No PDF files found in {data_dir}")

    return documents
