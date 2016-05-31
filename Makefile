gender: gender.cpp
	g++ gender.cpp -o gender -g -I/search/data/user/liukuang/py-faster-rcnn/caffe-fast-rcnn/include/ \
	  -I/usr/local/cuda-7.5/include/ \
	  -I/search/data/user/liukuang/py-faster-rcnn/caffe-fast-rcnn/build/src/ \
	  -I/usr/include/  \
	  -I/search/odin/caffe-deps/include/ \
	  -I/search/data/odin/caffe-deps/include/ImageMagick/ \
	  -L/search/odin/caffe-deps/lib/ \
	  -L/usr/lib64/  \
	  -L/search/data/user/liukuang/py-faster-rcnn/caffe-fast-rcnn/build/lib/ \
	  -L/search/data/user/liukuang/caffe/build/lib/ \
	  `pkg-config --cflags --libs opencv`  \
    	-lcaffe \
		-lMagick++   \
        -lMagickCore   \
        -lMagickWand 
clean:
	rm -rf gender
