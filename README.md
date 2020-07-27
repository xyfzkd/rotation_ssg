# rotation in Shilab
------

> **`relion ver3.1 src/acc/cuda` 优化`backproject`的kernel**

## preprocessing
```bash
relion_import --i "Movies/*.tiff" \
				 --odir Import/job001/ \
				 --ofile movies.star \
				 --do_movies true \
				 --optics_group_name opticsGroup1 \
				 --optics_group_mtf mtf_k2_200kV.star \
				 --angpix 0.885 \
				 --kV 200 \
				 --Cs 1.4 \
				 --Q0 0.1 
				 
			
relion_import  --do_movies \
				  --optics_group_name "opticsGroup1" \
				  --optics_group_mtf mtf_k2_200kV.star \
				  --angpix 0.885 \
				  --kV 200 \
				  --Cs 1.4 \
				  --Q0 0.1 \
				  --beamtilt_x 0 \
				  --beamtilt_y 0 \
				  --i "Movies/*.tiff" \
				  --odir Import/job001/ \
				  --ofile movies.star \
				  --odir Import/job001/
				 
```
!!! warning
    千万不要在家中尝试！