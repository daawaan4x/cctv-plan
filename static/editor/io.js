/// <reference path="./types.js" />
// @ts-check

{
  /** @type {EditorWindow} */
  const editorWindow = /** @type {EditorWindow} */ (window);
  editorWindow.EditorApp = /** @type {EditorAppNamespace} */ (editorWindow.EditorApp || {});
  const EditorApp = editorWindow.EditorApp;
  const {
    COLS,
    ROWS,
    STATE_BLACK,
    STATE_TRANSPARENT,
    STATE_WHITE,
  } = EditorApp;

  /**
   * Raw RGBA image data extracted through an offscreen canvas.
   * @typedef {{
   *   width: number,
   *   height: number,
   *   data: Uint8ClampedArray,
   * }} RgbaImageData
   */

  // Import helpers

  /**
   * Reads a selected file into an `HTMLImageElement`.
   * @param {File} file - User-selected image file.
   * @returns {Promise<HTMLImageElement>}
   */
  function loadImageViaFileReader(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('File could not be read.'));
      reader.onload = () => {
        const dataUrl = reader.result;
        if (typeof dataUrl !== 'string') {
          reject(new Error('File could not be converted into an image URL.'));
          return;
        }

        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('File was read, but the browser could not decode it as an image.'));
        img.src = dataUrl;
      };
      reader.readAsDataURL(file);
    });
  }

  /**
   * Decodes a grid image into logical cell states.
   * @param {HTMLImageElement} img - Decoded source image.
   * @param {string} sourceLabel - Source filename or display label.
   * @returns {ImportedGrid}
   */
  function importGridCellsFromImage(img, sourceLabel) {
    const { width, height, data } = ioGetRgbaImageData(img);
    const scaleX = width / COLS;
    const scaleY = height / ROWS;

    if (!Number.isInteger(scaleX) || !Number.isInteger(scaleY) || scaleX !== scaleY) {
      throw new Error(
        `Grid image ${sourceLabel} must be ${COLS}×${ROWS} or an integer export scale of it; got ${width}×${height}.`,
      );
    }

    const blockSize = scaleX;
    const cells = new Uint8Array(COLS * ROWS);

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const sourceRow = row * blockSize;
        const sourceCol = col * blockSize;
        const expectedState = ioGetPixelStateAt(data, width, sourceRow, sourceCol);

        // A scaled export should contain one solid state per logical cell block.
        ioAssertUniformBlock(data, width, sourceRow, sourceCol, blockSize, expectedState, sourceLabel, row, col);
        cells[(row * COLS) + col] = expectedState;
      }
    }

    return { cells, width, height, scale: blockSize };
  }

  /**
   * Starts a PNG download for the provided canvas.
   * @param {HTMLCanvasElement} canvasToDownload - Fully rendered export canvas.
   * @returns {void}
   */
  function downloadCanvasAsPng(canvasToDownload) {
    canvasToDownload.toBlob((blob) => {
      if (!blob) {
        alert('PNG export failed.');
        return;
      }

      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      const stamp = new Date().toISOString().replace(/[:.]/g, '-');
      anchor.href = url;
      anchor.download = `grid-210x90-${stamp}.png`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    }, 'image/png');
  }

  /**
   * Reads raw RGBA pixels from an image through an offscreen canvas.
   * @param {HTMLImageElement} img - Decoded source image.
   * @returns {RgbaImageData}
   */
  function ioGetRgbaImageData(img) {
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    if (!width || !height) throw new Error('Imported image has no readable dimensions.');

    const offscreen = document.createElement('canvas');
    offscreen.width = width;
    offscreen.height = height;

    const offscreenContext = offscreen.getContext('2d', { willReadFrequently: true });
    if (!offscreenContext) throw new Error('The browser could not read pixel data from the imported image.');

    offscreenContext.clearRect(0, 0, width, height);
    offscreenContext.drawImage(img, 0, 0, width, height);
    return { width, height, data: offscreenContext.getImageData(0, 0, width, height).data };
  }

  /**
   * Resolves one source pixel into a logical grid state.
   * @param {Uint8ClampedArray} data - Flat RGBA pixel data.
   * @param {number} width - Source image width in pixels.
   * @param {number} row - Zero-based source row.
   * @param {number} col - Zero-based source column.
   * @returns {number}
   */
  function ioGetPixelStateAt(data, width, row, col) {
    const offset = ((row * width) + col) * 4;
    return ioStateFromImportedPixel(
      data[offset],
      data[offset + 1],
      data[offset + 2],
      data[offset + 3],
    );
  }

  /**
   * Converts one RGBA pixel into the editor's three-state palette.
   * @param {number} r - Red channel.
   * @param {number} g - Green channel.
   * @param {number} b - Blue channel.
   * @param {number} a - Alpha channel.
   * @returns {number}
   */
  function ioStateFromImportedPixel(r, g, b, a) {
    // Treat low-alpha pixels as transparent and quantize the rest by luminance
    // so slightly off-black or off-white imports still collapse cleanly.
    if (a < 128) return STATE_TRANSPARENT;

    const luminance = (0.2126 * r) + (0.7152 * g) + (0.0722 * b);
    return luminance >= 127.5 ? STATE_WHITE : STATE_BLACK;
  }

  /**
   * Verifies that one scaled cell block resolves to a single logical state.
   * @param {Uint8ClampedArray} data - Flat RGBA pixel data.
   * @param {number} width - Source image width in pixels.
   * @param {number} startRow - Zero-based top row of the block.
   * @param {number} startCol - Zero-based left column of the block.
   * @param {number} blockSize - Width and height of the scaled block in pixels.
   * @param {number} expectedState - State inferred from the block's first pixel.
   * @param {string} sourceLabel - Source filename or display label.
   * @param {number} row - Zero-based logical grid row.
   * @param {number} col - Zero-based logical grid column.
   * @returns {void}
   */
  function ioAssertUniformBlock(data, width, startRow, startCol, blockSize, expectedState, sourceLabel, row, col) {
    for (let y = 0; y < blockSize; y++) {
      for (let x = 0; x < blockSize; x++) {
        const pixelState = ioGetPixelStateAt(data, width, startRow + y, startCol + x);
        if (pixelState === expectedState) continue;

        // Reject mixed blocks instead of guessing, otherwise anti-aliased or
        // partially edited exports could import inconsistently.
        throw new Error(
          `Grid image ${sourceLabel} mixes multiple states inside source cell block at row ${row}, col ${col}.`,
        );
      }
    }
  }

  EditorApp.loadImageViaFileReader = loadImageViaFileReader;
  EditorApp.importGridCellsFromImage = importGridCellsFromImage;
  EditorApp.downloadCanvasAsPng = downloadCanvasAsPng;
}
