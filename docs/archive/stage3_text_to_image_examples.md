# Stage 3 Retrieval Examples

Qualitative report-to-image retrieval examples from the OpenI split.

- Checkpoint: `checkpoints/best.pt`
- Split: `test`
- Direction: `text-to-image`
- Top-k: `5`
- Recall@5: `11.87%`

## Case 1: Query UID 120

- Query index: `5`
- Ground-truth full rank: `1`
- Ground-truth rank in top-5: `1`

**Query report**

Low lung volumes bilaterally with central bronchovascular crowding without focal consolidation, pleural effusion, or pneumothoraces.. Cardiomediastinal silhouette is within normal limits. Degenerative changes of the thoracic spine.. Low lung volumes bilaterally with central bronchovascular crowding without focal cardiopulmonary disease. .

**Retrieved images**

### Rank 1: UID 120 (MATCH)

- Similarity: `0.5190`
- Dataset index: `5`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/120_IM-0133-1001.dcm.png`

Low lung volumes bilaterally with central bronchovascular crowding without focal consolidation, pleural effusion, or pneumothoraces.. Cardiomediastinal silhouette is within normal limits. Degenerative changes of the thoracic spine.. Low lung volumes bilaterally with central bronchovascular crowding without focal cardiopulmonary disease. .

### Rank 2: UID 3882 (MISMATCH)

- Similarity: `0.4470`
- Dataset index: `308`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3882_IM-1970-1001.dcm.png`

Eventration of the left diaphragm is noted. Question left basilar atelectasis versus infiltrate. No evidence of pneumothorax. Generalized lung volumes. No definite pleural effusions. Heart size within normal limits. Osseous structures intact. Generalized low lung lungs with eventration of the left hemidiaphragm. Question concomitant left basilar opacity, may represent atelectasis or infiltrate.

### Rank 3: UID 1020 (MISMATCH)

- Similarity: `0.4426`
- Dataset index: `70`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1020_IM-0017-1001.dcm.png`

Lung volumes are low. No focal infiltrates. Heart size normal. Hypoinflation with no visible active cardiopulmonary disease.

### Rank 4: UID 3885 (MISMATCH)

- Similarity: `0.4220`
- Dataset index: `309`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3885_IM-1971-1001.dcm.png`

The heart, pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is no focal air space opacity to suggest a pneumonia. No acute cardiopulmonary disease.

### Rank 5: UID 1501 (MISMATCH)

- Similarity: `0.4210`
- Dataset index: `107`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1501_IM-0327-1001.dcm.png`

Chest. Right hemidiaphragm remains elevated. Consolidation and atelectasis are present in the right lung base. Left lung is clear. No pleural air collections. Shoulder and clavicle. Fractures present in the right scapula the base of the glenoid process. It is attached to the coracoid process and a portion of the spine. The humeral head is located within the glenoid articular surface. Cutaneous air is present. Fracture is present in the posterior portion of the right 3rd rib. The acromioclavicular joint and coracoclavicular joints are widened. 1. Chest. Continued right hemidiaphragm elevation with right lower lobe airspace disease. 2. Right sh...

## Case 2: Query UID 1261

- Query index: `87`
- Ground-truth full rank: `4`
- Ground-truth rank in top-5: `4`

**Query report**

Mild cardiomegaly. Normal pulmonary vascularity. Tortuosity of the descending aorta. No focal infiltrate, pneumothorax or pleural effusion. Mild cardiomegaly.

**Retrieved images**

### Rank 1: UID 3766 (MISMATCH)

- Similarity: `0.4782`
- Dataset index: `298`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3766_IM-1885-1001.dcm.png`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart is not significantly enlarged. There are atherosclerotic changes of the aorta. There are severe arthritic changes of the XXXX with mild arthritic changes of the thoracic spine. No acute pulmonary disease.

### Rank 2: UID 3423 (MISMATCH)

- Similarity: `0.4541`
- Dataset index: `267`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3423_IM-1656-1001.dcm.png`

the heart size is normal. There is tortuosity of aorta. Pulmonary vascularity is normal. No focal airspace disease or effusion. Degenerative changes in the thoracic spine. Tortuous aorta, otherwise unremarkable exam.

### Rank 3: UID 3014 (MISMATCH)

- Similarity: `0.4528`
- Dataset index: `237`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3014_IM-1392-1001.dcm.png`

There is distortion of the right hilum which may be postsurgical versus neoplastic. Volume loss of the right hand side. There is no evidence of focal infiltrate. No pneumothorax. No pleural effusion. Normal heart size. Question prior right upper lobe resection, no acute abnormality.

### Rank 4: UID 1261 (MATCH)

- Similarity: `0.4337`
- Dataset index: `87`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1261_IM-0177-1001.dcm.png`

Mild cardiomegaly. Normal pulmonary vascularity. Tortuosity of the descending aorta. No focal infiltrate, pneumothorax or pleural effusion. Mild cardiomegaly.

### Rank 5: UID 2979 (MISMATCH)

- Similarity: `0.4194`
- Dataset index: `235`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2979_IM-1368-1001-0001.dcm.png`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart is not significantly enlarged. There are calcified mediastinal lymph XXXX. There are atherosclerotic changes of the aorta. Arthritic changes of the skeletal structures are noted. No acute pulmonary disease.

## Case 3: Query UID 2439

- Query index: `192`
- Ground-truth full rank: `3`
- Ground-truth rank in top-5: `3`

**Query report**

The heart is normal in size. The mediastinum is unremarkable. The lungs are hyperinflated. There is biapical scarring. No acute infiltrate or pleural effusion seen. Emphysema without acute disease.

**Retrieved images**

### Rank 1: UID 965 (MISMATCH)

- Similarity: `0.5237`
- Dataset index: `66`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/965_IM-2455-1001.dcm.png`

The lungs appear clear. Heart and pulmonary XXXX appear normal. Mediastinal contours are normal. Pleural spaces are clear. There appears to the contrast XXXX within small colonic diverticula in the splenic flexure region. 1. No acute cardiopulmonary disease

### Rank 2: UID 2081 (MISMATCH)

- Similarity: `0.4973`
- Dataset index: `161`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2081_IM-0713-1001.dcm.png`

The lungs are well-expanded and clear. No pleural effusion or pneumothorax is seen. The cardiomediastinal contour is normal. No acute osseous lesions are identified. No active pulmonary disease.

### Rank 3: UID 2439 (MATCH)

- Similarity: `0.4861`
- Dataset index: `192`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2439_IM-0978-1001.dcm.png`

The heart is normal in size. The mediastinum is unremarkable. The lungs are hyperinflated. There is biapical scarring. No acute infiltrate or pleural effusion seen. Emphysema without acute disease.

### Rank 4: UID 1417 (MISMATCH)

- Similarity: `0.4788`
- Dataset index: `97`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1417_IM-0266-1001.dcm.png`

Cardiac and mediastinal contours are unremarkable. Pulmonary vascularity is within normal limits. No focal air space opacities, pleural effusion, or pneumothorax. There are increased lucencies in the bilateral apices along with horizontal oblique scarring in the left upper lobe. This could suggest emphysematous bullae. XXXX are grossly unremarkable. 1. No active disease.

### Rank 5: UID 655 (MISMATCH)

- Similarity: `0.4744`
- Dataset index: `42`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/655_IM-2231-1001.dcm.png`

Normal heart size and mediastinal contours. The lungs are hyperinflated but clear. No pneumothorax or pleural effusion. No acute bony abnormalities. No acute cardiopulmonary process. .

## Case 4: Query UID 3952

- Query index: `316`
- Ground-truth full rank: `5`
- Ground-truth rank in top-5: `5`

**Query report**

Unchanged elevation of the right hemidiaphragm. The trachea is midline. Negative for pneumothorax, pleural effusion or focal airspace consolidation. The heart size is mildly enlarged. Mild degenerative changes throughout the thoracic spine anterior osteophytes noted inferiorly. Pulmonary artery prominence. 1. Mild cardiomegaly. No acute cardiopulmonary abnormality.

**Retrieved images**

### Rank 1: UID 3880 (MISMATCH)

- Similarity: `0.4425`
- Dataset index: `307`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3880_IM-1968-1001.dcm.png`

Unchanged cardiomegaly. Negative for pneumothorax or focal consolidation. No large effusion. Mildly prominent interstitial opacities. Stable cardiomegaly with mild pulmonary interstitial edema.

### Rank 2: UID 2186 (MISMATCH)

- Similarity: `0.4310`
- Dataset index: `173`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2186_IM-0796-1001.dcm.png`

The lungs appear clear. Lung volumes are low. The heart and pulmonary XXXX appear normal. Pleural spaces are clear. No acute cardiopulmonary disease.

### Rank 3: UID 3442 (MISMATCH)

- Similarity: `0.4227`
- Dataset index: `269`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3442_IM-1667-1001.dcm.png`

Normal cardiomediastinal contours. Low lung volumes with minimal left basilar opacities. No pneumothorax or pleural effusions. Minimal left basilar atelectasis versus infiltrate. Low lung volumes.

### Rank 4: UID 1261 (MISMATCH)

- Similarity: `0.4188`
- Dataset index: `87`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1261_IM-0177-1001.dcm.png`

Mild cardiomegaly. Normal pulmonary vascularity. Tortuosity of the descending aorta. No focal infiltrate, pneumothorax or pleural effusion. Mild cardiomegaly.

### Rank 5: UID 3952 (MATCH)

- Similarity: `0.4108`
- Dataset index: `316`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3952_IM-2020-1001.dcm.png`

Unchanged elevation of the right hemidiaphragm. The trachea is midline. Negative for pneumothorax, pleural effusion or focal airspace consolidation. The heart size is mildly enlarged. Mild degenerative changes throughout the thoracic spine anterior osteophytes noted inferiorly. Pulmonary artery prominence. 1. Mild cardiomegaly. No acute cardiopulmonary abnormality.

## Case 5: Query UID 2

- Query index: `0`
- Ground-truth full rank: `100`
- Ground-truth rank in top-5: `not retrieved`

**Query report**

Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries. Clear lungs. Inferior XXXX XXXX XXXX. No acute pulmonary findings.

**Retrieved images**

### Rank 1: UID 3766 (MISMATCH)

- Similarity: `0.5293`
- Dataset index: `298`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3766_IM-1885-1001.dcm.png`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart is not significantly enlarged. There are atherosclerotic changes of the aorta. There are severe arthritic changes of the XXXX with mild arthritic changes of the thoracic spine. No acute pulmonary disease.

### Rank 2: UID 1829 (MISMATCH)

- Similarity: `0.5001`
- Dataset index: `134`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1829_IM-0537-1001.dcm.png`

XXXX XXXX and lateral chest examination was obtained. The heart silhouette and mediastinal contours are not enlarged. Removal of 2 left-sided chest tubes. There is no pneumothorax. Lungs demonstrate no acute findings. There is minimal posterior pleural effusions. 1. No pneumothorax following removal of left-sided chest tubes.

### Rank 3: UID 152 (MISMATCH)

- Similarity: `0.4834`
- Dataset index: `7`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/152_IM-0335-2001.dcm.png`

Stable cardiomediastinal silhouette with mild cardiomegaly and aortic ectasia and tortuosity. No alveolar consolidation, no findings of pleural effusion. Chronic appearing bilateral rib contour deformities compatible with old fractures. No pneumothorax. No acute findings.

### Rank 4: UID 3423 (MISMATCH)

- Similarity: `0.4777`
- Dataset index: `267`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3423_IM-1656-1001.dcm.png`

the heart size is normal. There is tortuosity of aorta. Pulmonary vascularity is normal. No focal airspace disease or effusion. Degenerative changes in the thoracic spine. Tortuous aorta, otherwise unremarkable exam.

### Rank 5: UID 3571 (MISMATCH)

- Similarity: `0.4614`
- Dataset index: `281`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3571_IM-1754-3001.dcm.png`

Heart size and pulmonary vascularity normal. The stomach contour normal. There is right hemidiaphragm elevation. Lungs are clear. Degenerative changes in the thoracic spine. Right hemidiaphragm elevation. No acute cardiopulmonary process.

## Case 6: Query UID 1555

- Query index: `110`
- Ground-truth full rank: `241`
- Ground-truth rank in top-5: `not retrieved`

**Query report**

XXXX XXXX and lateral chest examination was obtained. The heart silhouette and mediastinal contours are not enlarged. There is elevated right hemidiaphragm and evidence of right upper lobectomy. Lungs demonstrate no acute findings. There is no effusion or pneumothorax. 1. No acute cardiopulmonary disease.

**Retrieved images**

### Rank 1: UID 1339 (MISMATCH)

- Similarity: `0.3721`
- Dataset index: `92`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1339_IM-0218-1001.dcm.png`

Small 3.3 mm right-sided pneumothorax only visible on the left lateral decubitus film. Left lung is clear. Normal cardiac contour. No evidence of pleural effusion. 1. Small 3.3 mm right-sided pneumothorax.

### Rank 2: UID 3525 (MISMATCH)

- Similarity: `0.3348`
- Dataset index: `277`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3525_IM-1722-6001.dcm.png`

Heart size normal. No pneumothorax, pleural effusion, or focal airspace disease. The visualized bony structures appear intact. There is a XXXX radiodensity overlying the right shoulder which is XXXX external to the patient however clinical correlation recommended. Scattered calcified granulomas. No acute cardiopulmonary abnormality. No fracture visualized.

### Rank 3: UID 3926 (MISMATCH)

- Similarity: `0.3324`
- Dataset index: `315`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3926_IM-2000-4004.dcm.png`

The trachea is midline. The heart XXXX is large, unchanged from prior exam. Slightly widened mediastinum, secondary to cardiomegaly and a tortuous aorta, is accentuated by AP portable technique. There are low lung volumes causing bibasilar atelectasis and bronchovascular crowding. The lungs do not demonstrate focal infiltrate or effusion. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities. 1. Low volume study without acute cardiopulmonary abnormalities. .

### Rank 4: UID 487 (MISMATCH)

- Similarity: `0.3313`
- Dataset index: `32`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/487_IM-2110-1001.dcm.png`

Cardiomediastinal silhouettes are within normal limits. Lungs are clear without focal consolidation, pneumothorax, or pleural effusion. Bony thorax is unremarkable. No acute cardiopulmonary abnormalities.

### Rank 5: UID 1074 (MISMATCH)

- Similarity: `0.3201`
- Dataset index: `77`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1074_IM-0054-1001.dcm.png`

The cardiomediastinal silhouette is stable. Lung volumes remain low. There is no pleural line to suggest pneumothorax or costophrenic XXXX blunting to suggest large pleural effusion. Bony structures are within normal limits. Low lung volumes. No acute cardiopulmonary findings.

## Case 7: Query UID 2754

- Query index: `215`
- Ground-truth full rank: `21`
- Ground-truth rank in top-5: `not retrieved`

**Query report**

The lungs are clear. The heart and pulmonary XXXX are normal. The pleural spaces are clear. Mediastinal contours are normal. No acute cardiopulmonary disease

**Retrieved images**

### Rank 1: UID 1833 (MISMATCH)

- Similarity: `0.3833`
- Dataset index: `136`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1833_IM-0539-1001.dcm.png`

There is a small area of scarring or atelectasis in the left base. Calcified granulomas seen in the posterior right lower lobe. Lungs are otherwise clear. The heart and mediastinum are normal. The skeletal structures and soft tissues are normal. Minimal small area scarring of the left base.

### Rank 2: UID 965 (MISMATCH)

- Similarity: `0.3779`
- Dataset index: `66`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/965_IM-2455-1001.dcm.png`

The lungs appear clear. Heart and pulmonary XXXX appear normal. Mediastinal contours are normal. Pleural spaces are clear. There appears to the contrast XXXX within small colonic diverticula in the splenic flexure region. 1. No acute cardiopulmonary disease

### Rank 3: UID 474 (MISMATCH)

- Similarity: `0.3724`
- Dataset index: `29`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/474_IM-2101-1001.dcm.png`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

### Rank 4: UID 1545 (MISMATCH)

- Similarity: `0.3699`
- Dataset index: `109`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1545_IM-0355-2002.dcm.png`

No focal areas of consolidation. No pneumothorax. Heart size within normal limits. No pleural effusions. Osseous structures intact. No acute cardiopulmonary abnormality. .

### Rank 5: UID 76 (MISMATCH)

- Similarity: `0.3677`
- Dataset index: `3`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/76_IM-2309-1001.dcm.png`

Apparent scarring within the lingula. Lungs are otherwise clear. No pleural effusions or pneumothoraces. Heart and mediastinum of normal size and contour. Apparent scarring within the lingula, otherwise unremarkable.

## Case 8: Query UID 3981

- Query index: `319`
- Ground-truth full rank: `10`
- Ground-truth rank in top-5: `not retrieved`

**Query report**

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

**Retrieved images**

### Rank 1: UID 2297 (MISMATCH)

- Similarity: `0.4754`
- Dataset index: `182`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2297_IM-0877-1001.dcm.png`

The cardiac contours are normal. The lungs are clear. Thoracic spondylosis. Mild XXXX XXXX curvature thoracolumbar junction. No active pulmonary disease.

### Rank 2: UID 3480 (MISMATCH)

- Similarity: `0.4518`
- Dataset index: `272`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3480_IM-1691-1001.dcm.png`

The cardiomediastinal silhouette and pulmonary vasculature are within normal limits. There is no pneumothorax or pleural effusion. There are no focal areas of consolidation. Cholecystectomy clips are present. No acute cardiopulmonary abnormality.

### Rank 3: UID 939 (MISMATCH)

- Similarity: `0.4516`
- Dataset index: `62`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/939_IM-2435-1001.dcm.png`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

### Rank 4: UID 2091 (MISMATCH)

- Similarity: `0.4432`
- Dataset index: `163`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2091_IM-0722-1001.dcm.png`

The cardiac contours are normal. The lungs are clear. Thoracic spondylosis. XXXX XXXX of the spine. No acute process.

### Rank 5: UID 2350 (MISMATCH)

- Similarity: `0.4356`
- Dataset index: `186`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2350_IM-0916-1001.dcm.png`

No focal areas of consolidation. No suspicious pulmonary opacities. Mild degenerative change thoracic spine. No pleural effusions. No evidence of pneumothorax. Heart size normal limits. No acute cardiopulmonary abnormality. .

