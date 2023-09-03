# identify-ship
Use deep-learning to dentify ships based on pictures taken from harbour (or another ship). This is only a proof of concept, trained on a total of 759 images from 53 ships around Oslo.

## Dataset
All ships with an IMO identifier that were within 50nm of Oslo on the 18th of August 2023. I scraped the first 30 images from a Bing search of an IMO number using [this repo](https://github.com/ultralytics/google-images-download), then manually cleaned by removing the images that:
1) Didn't show a ship
2) Included two or more ships, and the ship in question wasn't the main focus of the image
3) Obviously showed the wrong ship. This was tricky, because ships can be repainted or otherwise change appearance. When in doubt, the image wasn't removed.

All ships with less than 7 images were removed from the dataset. The final dataset contains 53 classes (unique ships) with an average of 14 images (range:7-30). Three images from each class were chosen at random to form the test dataset (total n=159), one from each class for validation.
The initial list of ships around Oslo was exported using the filter function in [Marine Traffic](https://www.marinetraffic.com).

## Approach
Pretrained EfficientNetb0, only the last layer was retrained. Regularized with quite strong data augmentation, dropout and weight decay. Pretty much no hyperparameter tuning, 'cause who's got time for that... (so definitely possible to improve on this!)

## Results
54% top-1 and 81% top-5 accuracy. Meaning that in 81% of cases, the right ship is found in the top 5 predictions of the model. Some correct (green) and incorrect (red) top-1 predictions:


![alt text](https://github.com/lidialuq/identify-ship/blob/main/results/examples.png)

Some ships have "twins" in the dataset, separate ships of the same ship model. Since the ships look the same, neither the model or I manage to classify them correctly (lower image, first column).
