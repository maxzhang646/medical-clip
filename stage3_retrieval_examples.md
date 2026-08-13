# Stage 3 Retrieval Examples

Qualitative image-to-report retrieval examples from the OpenI split.

- Checkpoint: `checkpoints/best.pt`
- Split: `test`
- Top-k: `5`
- Recall@5: `12.50%`

## Case 1: Query UID 120

- Query index: `5`
- Ground-truth full rank: `5`
- Ground-truth rank in top-5: `5`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/120_IM-0133-1001.dcm.png`

**Ground-truth report**

Low lung volumes bilaterally with central bronchovascular crowding without focal consolidation, pleural effusion, or pneumothoraces.. Cardiomediastinal silhouette is within normal limits. Degenerative changes of the thoracic spine.. Low lung volumes bilaterally with central bronchovascular crowding without focal cardiopulmonary disease. .

**Retrieved reports**

### Rank 1: UID 1169 (MISMATCH)

- Similarity: `0.5552`
- Dataset index: `79`

Lung volumes are low. There is vague opacity in the right upper lung near the anterior right first rib on PA view. This may be artifact relating to calcification at the first rib costicartilage junction. There is minimal atelectasis in the right lung base. There is left-sided PICC line, the distal tip in the lower superior vena XXXX. The heart and pulmonary XXXX are normal. These contours are normal. 1. Vague nodular opacity near the anterior right first rib costicartilage junction. This may be calcification. 2. Minimal streaky atelectasis in the right lung base.

### Rank 2: UID 828 (MISMATCH)

- Similarity: `0.5334`
- Dataset index: `56`

The trachea is midline. The cardiomediastinal silhouette is normal. There are low lung volumes, causing bibasilar atelectasis and bronchovascular crowding. There is a XXXX opacity in the left lingula. There is no pleural effusion or pneumothorax. Visualized bony structures reveal no acute abnormalities. 1. Low lung volumes. 2. Opacity in the lingula is favored to represent prominent pericardial fat, but lingular atelectasis or infiltrate cannot be ruled out. .

### Rank 3: UID 3442 (MISMATCH)

- Similarity: `0.5320`
- Dataset index: `269`

Normal cardiomediastinal contours. Low lung volumes with minimal left basilar opacities. No pneumothorax or pleural effusions. Minimal left basilar atelectasis versus infiltrate. Low lung volumes.

### Rank 4: UID 1591 (MISMATCH)

- Similarity: `0.5272`
- Dataset index: `116`

The trachea is midline. The cardio mediastinal silhouette is of normal size and contour. No evidence of focal infiltrate or effusion. Low lung volumes XXXX XXXX atelectasis and bronchovascular crowding. There is no pneumothorax. The visualized bony structures reveal no acute abnormalities. Lateral view reveals degenerative changes of the thoracic spine. 1. No acute cardiopulmonary abnormalities. 2. Low lung volumes causing bibasilar atelectasis and bronchovascular crowding .

### Rank 5: UID 120 (MATCH)

- Similarity: `0.5190`
- Dataset index: `5`

Low lung volumes bilaterally with central bronchovascular crowding without focal consolidation, pleural effusion, or pneumothoraces.. Cardiomediastinal silhouette is within normal limits. Degenerative changes of the thoracic spine.. Low lung volumes bilaterally with central bronchovascular crowding without focal cardiopulmonary disease. .

## Case 2: Query UID 1459

- Query index: `104`
- Ground-truth full rank: `2`
- Ground-truth rank in top-5: `2`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1459_IM-0297-1001.dcm.png`

**Ground-truth report**

No stable cardiomegaly, without focal consolidation, pneumothorax, or pleural effusion. Stable right basilar calcified granuloma. No acute osseous abnormality identified. Stable cardiomegaly without acute cardiopulmonary abnormality.

**Retrieved reports**

### Rank 1: UID 3379 (MISMATCH)

- Similarity: `0.4001`
- Dataset index: `265`

No pneumothorax or large pleural effusion. Borderline cardiomegaly. Minimal retrocardiac airspace disease. Bony structures appear intact. Bony structures appear intact. Minimal retrocardiac airspace disease.

### Rank 2: UID 1459 (MATCH)

- Similarity: `0.3904`
- Dataset index: `104`

No stable cardiomegaly, without focal consolidation, pneumothorax, or pleural effusion. Stable right basilar calcified granuloma. No acute osseous abnormality identified. Stable cardiomegaly without acute cardiopulmonary abnormality.

### Rank 3: UID 2246 (MISMATCH)

- Similarity: `0.3892`
- Dataset index: `179`

Normal heart size and mediastinal contour. Right lung base airspace disease on frontal XXXX. XXXX opacities in the left lung base consistent with atelectasis. No pneumothorax. No pleural effusion. Mild wedge XXXX deformity of T12. Right lung base airspace disease and left base atelectasis.

### Rank 4: UID 916 (MISMATCH)

- Similarity: `0.3852`
- Dataset index: `60`

Lungs are clear without focal airspace disease. Numerous XXXX calcifications are again noted. No pleural effusions or pneumothoraces. heart size is upper limits of normal. Clear lungs with heart size upper limits of normal.

### Rank 5: UID 504 (MISMATCH)

- Similarity: `0.3793`
- Dataset index: `33`

Stable cardiomediastinal silhouette. Stable XXXX opacity in the left base, XXXX scarring or atelectasis. Rounded calcified density in the left lung base, XXXX calcified granuloma. No XXXX consolidation. No pleural effusion or pneumothorax. Stable degenerative changes of the spine. No acute cardiopulmonary abnormality.

## Case 3: Query UID 3006

- Query index: `236`
- Ground-truth full rank: `3`
- Ground-truth rank in top-5: `3`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3006_IM-1388-1001.dcm.png`

**Ground-truth report**

Cardiomegaly is present. The pulmonary vascularity appears within normal limits. Thoracic aorta is tortuous. Patient is status post XXXX sternotomy. Surgical clips are present in the left axilla. Lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. There is eventration of the right hemidiaphragm. Degenerative changes are present in the spine. No evidence of active disease.

**Retrieved reports**

### Rank 1: UID 2102 (MISMATCH)

- Similarity: `0.5484`
- Dataset index: `164`

Tortuosity of the aorta. No pneumothorax, pleural effusion or airspace consolidation. Cardiomediastinal size is within normal limits. Pulmonary vasculature is normal . XXXX XXXX intact. Unchanged eventration of the left hemidiaphragm versus small hernia (Bochdalek). No acute cardiopulmonary abnormality. .

### Rank 2: UID 1524 (MISMATCH)

- Similarity: `0.5041`
- Dataset index: `108`

Heart size is normal. The aorta is tortuous, and cannot exclude ascending aortic aneurysm. The pulmonary vascularity is normal. There residual to prior granulomatous infection. Lungs are otherwise clear. Degenerative change of the spine. 1. No acute cardiopulmonary process. 2. Tortuous aorta, cannot exclude ascending aortic aneurysm.

### Rank 3: UID 3006 (MATCH)

- Similarity: `0.4898`
- Dataset index: `236`

Cardiomegaly is present. The pulmonary vascularity appears within normal limits. Thoracic aorta is tortuous. Patient is status post XXXX sternotomy. Surgical clips are present in the left axilla. Lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. There is eventration of the right hemidiaphragm. Degenerative changes are present in the spine. No evidence of active disease.

### Rank 4: UID 1405 (MISMATCH)

- Similarity: `0.4692`
- Dataset index: `96`

There is scarring in the right mid and upper lung zone with surgical clips identified as well. There is no pleural effusion or pneumothorax. The heart is not significantly enlarged. There are atherosclerotic changes of the aorta. Arthritic changes of the skeletal structures are noted. No acute pulmonary disease.

### Rank 5: UID 2888 (MISMATCH)

- Similarity: `0.4691`
- Dataset index: `227`

Heart size and pulmonary vascularity appear within normal limits. Descending thoracic aorta is tortuous. Lungs are free of focal airspace disease. No pleural effusion or pneumothorax is seen. Degenerative changes are present in the spine. No evidence of active disease.

## Case 4: Query UID 3981

- Query index: `319`
- Ground-truth full rank: `5`
- Ground-truth rank in top-5: `5`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3981_IM-2039-1001.dcm.png`

**Ground-truth report**

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

**Retrieved reports**

### Rank 1: UID 3571 (MISMATCH)

- Similarity: `0.4123`
- Dataset index: `281`

Heart size and pulmonary vascularity normal. The stomach contour normal. There is right hemidiaphragm elevation. Lungs are clear. Degenerative changes in the thoracic spine. Right hemidiaphragm elevation. No acute cardiopulmonary process.

### Rank 2: UID 474 (MISMATCH)

- Similarity: `0.4040`
- Dataset index: `29`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

### Rank 3: UID 3981 (MATCH)

- Similarity: `0.4040`
- Dataset index: `319`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

### Rank 4: UID 939 (MISMATCH)

- Similarity: `0.4040`
- Dataset index: `62`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

### Rank 5: UID 246 (MISMATCH)

- Similarity: `0.4040`
- Dataset index: `14`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

## Case 5: Query UID 2

- Query index: `0`
- Ground-truth full rank: `184`
- Ground-truth rank in top-5: `not retrieved`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2_IM-0652-1001.dcm.png`

**Ground-truth report**

Borderline cardiomegaly. Midline sternotomy XXXX. Enlarged pulmonary arteries. Clear lungs. Inferior XXXX XXXX XXXX. No acute pulmonary findings.

**Retrieved reports**

### Rank 1: UID 3571 (MISMATCH)

- Similarity: `0.4767`
- Dataset index: `281`

Heart size and pulmonary vascularity normal. The stomach contour normal. There is right hemidiaphragm elevation. Lungs are clear. Degenerative changes in the thoracic spine. Right hemidiaphragm elevation. No acute cardiopulmonary process.

### Rank 2: UID 2363 (MISMATCH)

- Similarity: `0.4721`
- Dataset index: `187`

Heart size is upper limits of normal. The pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is right basilar air space opacity. Right middle lobe and lower lobe pneumonia. Followup radiographs in 8-12 weeks after appropriate therapy are indicated to exclude an underlying abnormality.

### Rank 3: UID 3016 (MISMATCH)

- Similarity: `0.4596`
- Dataset index: `238`

Cardiac and mediastinal contours are within normal limits. Prior granulomatous disease. The lungs are clear. Thoracic spondylosis. No acute findings.

### Rank 4: UID 3603 (MISMATCH)

- Similarity: `0.4265`
- Dataset index: `284`

Cardiac and mediastinal contours are within normal limits. The lungs are clear. Bony structures are intact. No acute preoperative findings

### Rank 5: UID 939 (MISMATCH)

- Similarity: `0.4053`
- Dataset index: `62`

The lungs are clear. There is no pleural effusion or pneumothorax. The heart and mediastinum are normal. The skeletal structures are normal. No acute pulmonary disease.

## Case 6: Query UID 1501

- Query index: `107`
- Ground-truth full rank: `56`
- Ground-truth rank in top-5: `not retrieved`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/1501_IM-0327-1001.dcm.png`

**Ground-truth report**

Chest. Right hemidiaphragm remains elevated. Consolidation and atelectasis are present in the right lung base. Left lung is clear. No pleural air collections. Shoulder and clavicle. Fractures present in the right scapula the base of the glenoid process. It is attached to the coracoid process and a portion of the spine. The humeral head is located within the glenoid articular surface. Cutaneous air is present. Fracture is present in the posterior portion of the right 3rd rib. The acromioclavicular joint and coracoclavicular joints are widened. 1. Chest. Continued right hemidiaphragm elevation with right lower lobe airspace disease. 2. Right sh...

**Retrieved reports**

### Rank 1: UID 2177 (MISMATCH)

- Similarity: `0.5433`
- Dataset index: `172`

XXXX XXXX and lateral chest examination was obtained. There is enlarged heart silhouette. Decreased lung volumes. Lungs demonstrate bibasilar airspace opacities better visualized on lateral view. There is no effusion or pneumothorax. Degenerative changes of the bilateral XXXX. 1. Decreased lung volumes. Bibasilar airspace opacities seen on lateral XXXX XXXX be atelectasis or possibly pneumonia.

### Rank 2: UID 3536 (MISMATCH)

- Similarity: `0.5371`
- Dataset index: `278`

Heart size is enlarged, pulmonary vascularity within normal limits. No visible pneumothorax . XXXX right pleural effusion blunting posterior costophrenic XXXX. There is a XXXX XXXX of subsegmental atelectasis of the left lung base. There is XXXX alveolar airspace disease in the medial right lung base. Multilevel degenerative disease of the visualized portions of the thoracolumbar spine. 1. Cardiomegaly without pulmonary edema. 2. XXXX right medial basilar airspace disease. 3. Left lower lobe subsegmental atelectasis.

### Rank 3: UID 2954 (MISMATCH)

- Similarity: `0.5025`
- Dataset index: `234`

There is obscuration of the bilateral lung bases with lower lung volumes compared to prior examination. Stable atelectatic/fibrotic changes of the visualized lung, and stable left-sided calcified granuloma. No acute osseous abnormalities identified. Cardiomediastinal silhouette unremarkable. Obscuration of the bilateral lung bases, XXXX combination of atelectasis, infiltrate, effusions.

### Rank 4: UID 1832 (MISMATCH)

- Similarity: `0.5011`
- Dataset index: `135`

Interval removal of cardiac XXXX generator. Cardiomegaly. Left base streaky opacities again noted. No large focal areas of consolidation. No pleural effusions. Osseous structures intact. No pneumothorax. 1. Streaky left basilar opacities, XXXX atelectasis versus infiltrate. 2. Cardiomegaly, stable.

### Rank 5: UID 545 (MISMATCH)

- Similarity: `0.4944`
- Dataset index: `34`

Lung volumes are XXXX. XXXX opacities are present in both lung bases. A hiatal hernia is present. Heart and pulmonary XXXX are normal. Hypoinflation with bibasilar focal atelectasis.

## Case 7: Query UID 2718

- Query index: `212`
- Ground-truth full rank: `105`
- Ground-truth rank in top-5: `not retrieved`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/2718_IM-1182-1001.dcm.png`

**Ground-truth report**

The cardiomediastinal contours are within normal limits. Pulmonary vasculature is unremarkable. There is no focal airspace opacity. No pleural effusion or pneumothorax is seen. There are mild degenerative changes along the thoracic spine. No acute bony abnormality is identified. No acute cardiopulmonary abnormality.

**Retrieved reports**

### Rank 1: UID 3977 (MISMATCH)

- Similarity: `0.4672`
- Dataset index: `318`

Normal heart size. Stable unfolding the thoracic aorta. No focal air space consolidation. No pleural effusion or pneumothorax. Stable calcified granuloma in the left lower lobe. Visualized osseous structures are unremarkable appearance. No acute cardiopulmonary abnormality.

### Rank 2: UID 504 (MISMATCH)

- Similarity: `0.4401`
- Dataset index: `33`

Stable cardiomediastinal silhouette. Stable XXXX opacity in the left base, XXXX scarring or atelectasis. Rounded calcified density in the left lung base, XXXX calcified granuloma. No XXXX consolidation. No pleural effusion or pneumothorax. Stable degenerative changes of the spine. No acute cardiopulmonary abnormality.

### Rank 3: UID 593 (MISMATCH)

- Similarity: `0.4284`
- Dataset index: `40`

No acute cardiopulmonary abnormality. Extensive degenerative changes of the thoracic spine. Mildly enlarged heart. Tortuous aorta. Aortic calcifications. No focal area of consolidation, pleural effusion or pneumothorax. No acute radiographic cardiopulmonary process.

### Rank 4: UID 472 (MISMATCH)

- Similarity: `0.4089`
- Dataset index: `28`

Normal heart size and mediastinal contours. No focal airspace consolidation. No pleural effusion or pneumothorax. Stable postoperative and degenerative changes of the XXXX. Stable degenerative disc disease of the thoracic spine. No acute cardiopulmonary abnormalities.

### Rank 5: UID 400 (MISMATCH)

- Similarity: `0.4014`
- Dataset index: `24`

The lungs are clear bilaterally. Specifically, no evidence of focal consolidation, pneumothorax, or pleural effusion. Stable small right basilar calcified granuloma. Cardio mediastinal silhouette is unremarkable. Visualized osseous structures of the thorax are without acute abnormality. No acute cardiopulmonary abnormality.

## Case 8: Query UID 3977

- Query index: `318`
- Ground-truth full rank: `43`
- Ground-truth rank in top-5: `not retrieved`
- Image path: `/Users/wenluzhang/.cache/kagglehub/datasets/raddar/chest-xrays-indiana-university/versions/2/images/images_normalized/3977_IM-2036-1001.dcm.png`

**Ground-truth report**

Normal heart size. Stable unfolding the thoracic aorta. No focal air space consolidation. No pleural effusion or pneumothorax. Stable calcified granuloma in the left lower lobe. Visualized osseous structures are unremarkable appearance. No acute cardiopulmonary abnormality.

**Retrieved reports**

### Rank 1: UID 2142 (MISMATCH)

- Similarity: `0.3821`
- Dataset index: `169`

PA and lateral views the chest were obtained. The cardiomediastinal silhouette is normal in size and configuration. The lungs are well aerated. No pneumothorax, pleural effusion, or focal air space consolidation. Probable DISH of the thoracic spine. No acute cardiopulmonary disease.

### Rank 2: UID 2203 (MISMATCH)

- Similarity: `0.3777`
- Dataset index: `175`

PA and lateral views of the chest were obtained. The cardiomediastinal silhouette is normal in size and configuration. The lungs are well aerated. There is no pneumothorax, pleural effusion, or focal air space consolidation. Old right rib fractures. 1. No acute cardiopulmonary disease.

### Rank 3: UID 1847 (MISMATCH)

- Similarity: `0.3763`
- Dataset index: `138`

The cardiac and mediastinal silhouettes are unremarkable. The lungs are well expanded and clear. There is no focal air space opacity, pneumothorax, or effusion. There are large calcified mediastinal and right hilar granulomas. The bony structures of the thorax are intact with no evidence of acute abnormality. No evidence of acute cardiopulmonary process. Stable appearance of the chest.

### Rank 4: UID 709 (MISMATCH)

- Similarity: `0.3734`
- Dataset index: `45`

Heart size is normal. The lungs are clear. There are no focal air space consolidations. No pleural effusions or pneumothoraces. Aortic vascular calcifications. Normal pulmonary vascularity. Fracture-dislocation of the right shoulder. Bone demineralization. Scoliosis which is possibly positional. Clear lungs. Fracture-dislocation of the proximal right shoulder .

### Rank 5: UID 608 (MISMATCH)

- Similarity: `0.3720`
- Dataset index: `41`

The heart and mediastinum are unremarkable. The lungs are clear without infiltrate. There is no effusion or pneumothorax. There is an old healed fracture through the right 8th rib. 1. No acute cardiopulmonary disease.

