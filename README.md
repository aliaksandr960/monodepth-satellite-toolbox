# Monodepth Satallite Toolbox
*Functionality to process satallite imagery with Monocular Depth neural networks*

# Usage:
- *python pipeline.py 'path to reconstruction folder'*, reconstruction folder should have *'raster.tif'* file
- use jupyter-notebook and *pipeline.ipynb* file

Both files have dictionary with configuration, so you could adjust it.

Best use with Z18 scale, or about 0.6m GSD.

## This toolbox can:
1. **Split big GeoTIF image on patches, process each patch with monocular depth model, normalizer results and save as GeoTIF image**
![split infer merge](docs/split-infer-merge.jpg)

2. **Segment wall and сliffs**
![walls and cliffs](docs/walls-and-cliffs.jpg)

3. **Ortho from perspective images**
![ortho](docs/ortho.jpg)

4. **Occlusion map, to see what cannot be seen**
![occlusion](docs/occlusion_map.jpg)

## Algorithms description:

#### 1. split.py

- Grabs GeoTiff, split it to overlapping patches.

#### 2. depthmaps.py

- Perform monocular depth estimation for each patch using **Apple DepthPro** or **Meta Depth Anything** models **(configurable)**.

- Rellys on **HuggingFace** Transformers module, so it could be easy to integrate any model avaliable on **HuggingFace**.

#### 3. heightmaps.py

- Invert depthmaps

- Do min-pooling and smoothing to estimate background bias

- Substract background from inverted depth

*Could be a probles on really large buildings*

![Background](docs/remove_background.jpg)


#### 4. directions.py

- Slice inverted depth with high gradients by some number of levels

- Skeletonize and cross-correlate levels between each other

- Max cross correlation is view direction

![Cross correlation](docs/cross_corr.jpg)

#### 5. basic_analytics.py

- After having view direction, it is possible to estimate subvertical surfaces and normalize inverted depth maps

- Calculate walls and cliffs


#### 6. merge_analytics.py
- Overlapping patches merging using center-distance weighting to minimize visible differences between them


#### 7. ortho.py
- Convert analytics and raster to point cloud -> transform > store as color.tif and height.tif


## Tests

Due I not found any specific datasetes, I just grabbed 6 images form all around the world.

If ortho looks fine - it means height estimation and depthmaps also fine.

I used Meta Depth Anyting V2 model, becouse it is more CPU-friendly

![Cross correlation](docs/test_1.jpg)

Across all six samples, none failed completely. Directions were calculated quite accurately in every case. There were some issues, such as inaccurately segmented walls in Batumi and background removal problems for the Chicago warehouse. However, overall, everything looks reasonably good.

 # Licensing:
 - The code is released under the MIT License.
 - File *'test_reconstruction/raster.tif'* is a screenshot from Google Maps. Its usage should comply with Google Maps' Terms of Service."
 - Model weights and dependencies are licensed by their respective authors.