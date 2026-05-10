Compile to slideshow video with the following:

```
ffmpeg -framerate 1 -start_number 10 -i dori_heatmaps_k%d.png -vf "scale=1920:-2" -c:v libx264 -r 30 -pix_fmt yuv420p output.mp4
```
