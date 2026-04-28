/// <reference path="./types.js" />
// @ts-check

{
  /** @type {EditorWindow} */
  const editorWindow = /** @type {EditorWindow} */ (window);
  editorWindow.EditorApp = /** @type {EditorAppNamespace} */ (editorWindow.EditorApp || {});
  const EditorApp = editorWindow.EditorApp;
  const {
    COLS,
    MAX_ZOOM,
    MIN_ZOOM,
    PREVIEW_CELL,
    PREVIEW_H,
    PREVIEW_W,
    ROWS,
    STATE_TRANSPARENT,
    STATE_WHITE,
    VIEWPORT_PADDING,
    countCells,
    getGridImportStatus,
    getLineCells,
    getPaintTarget,
    getUnderlayStatus,
  } = EditorApp;

  /**
   * Rectangle returned by the contain-fit image layout helper.
   * @typedef {{
   *   drawX: number,
   *   drawY: number,
   *   drawW: number,
   *   drawH: number,
   * }} ContainRect
   */

  /**
   * Creates the rendering adapter for the editor canvas and related UI labels.
   * @param {EditorElements} elements - DOM references required for drawing and UI updates.
   * @returns {EditorRenderer}
   */
  function createEditorRenderer(elements) {
    const canvasContext = elements.canvas.getContext('2d', { alpha: true });
    if (!canvasContext) throw new Error('The editor canvas context could not be created.');
    /** @type {CanvasRenderingContext2D} */
    const ctx = canvasContext;

    elements.canvas.width = PREVIEW_W;
    elements.canvas.height = PREVIEW_H;

    /**
     * Sizes the visible canvas shell for the current zoom level.
     * @param {number} zoomScale - Logical zoom multiplier.
     * @returns {void}
     */
    function syncCanvasLayout(zoomScale) {
      const availableWidth = Math.max(160, elements.canvasScroll.clientWidth - VIEWPORT_PADDING - 1);
      const availableHeight = Math.max(120, elements.canvasScroll.clientHeight - VIEWPORT_PADDING - 1);
      const aspectRatio = PREVIEW_W / PREVIEW_H;
      const fitWidth = Math.min(availableWidth, availableHeight * aspectRatio);
      const zoomedWidth = Math.max(160, Math.floor(fitWidth * zoomScale));
      const zoomedHeight = Math.max(120, Math.floor(zoomedWidth / aspectRatio));

      // The scroll container owns the visible size while the canvas keeps a fixed
      // backing resolution, which keeps pointer math stable across zoom levels.
      elements.canvasShell.style.width = `${zoomedWidth}px`;
      elements.canvasShell.style.height = `${zoomedHeight}px`;
      elements.zoomValue.textContent = `${Math.round(zoomScale * 100)}%`;
      elements.zoomOutBtn.disabled = zoomScale <= MIN_ZOOM + 0.001;
      elements.zoomInBtn.disabled = zoomScale >= MAX_ZOOM - 0.001;
    }

    /**
     * Refreshes the editor readouts and repaints the preview canvas.
     * @param {EditorState} state - Current editor state.
     * @param {ViewState} view - Current view-only control values.
     * @param {string} cursorText - Cursor text already resolved by the controller.
     * @returns {void}
     */
    function refresh(state, view, cursorText) {
      updateStatus(state);
      updateHistoryButtons(state);
      updateSelections(state);
      updateSliderLabels(view);
      updateCursor(cursorText);
      renderCanvas(state, view);
    }

    /**
     * Renders an export-only canvas using the requested output settings.
     * @param {EditorState} state - Current editor state.
     * @param {ExportOptions} options - Export configuration chosen by the user.
     * @returns {HTMLCanvasElement}
     */
    function renderExportCanvas(state, options) {
      const output = document.createElement('canvas');
      output.width = COLS * options.scale;
      output.height = ROWS * options.scale;

      const out = output.getContext('2d', { alpha: true });
      if (!out) throw new Error('The export canvas context could not be created.');

      out.clearRect(0, 0, output.width, output.height);
      if (options.includeUnderlay === 'with-underlay') {
        rendererDrawUnderlay(out, state.underlayImage, output.width, output.height, options.underlayOpacity);
      }

      rendererDrawCells(out, state.cells, options.scale, 1);
      return output;
    }

    /**
     * Updates the status strings shown beside the editor controls.
     * @param {EditorState} state - Current editor state.
     * @returns {void}
     */
    function updateStatus(state) {
      const counts = countCells(state);
      elements.status.textContent = `Grid: ${COLS} columns × ${ROWS} rows. Transparent: ${counts.transparent.toLocaleString()} · White: ${counts.white.toLocaleString()} · Black: ${counts.black.toLocaleString()}.`;
      elements.underlayStatus.textContent = getUnderlayStatus(state);
      elements.gridImportStatus.textContent = getGridImportStatus(state);
    }

    /**
     * Syncs the undo and redo button disabled states.
     * @param {EditorState} state - Current editor state.
     * @returns {void}
     */
    function updateHistoryButtons(state) {
      elements.undoBtn.disabled = state.undoStack.length === 0;
      elements.redoBtn.disabled = state.redoStack.length === 0;
    }

    /**
     * Marks the selected swatch and tool button as active.
     * @param {EditorState} state - Current editor state.
     * @returns {void}
     */
    function updateSelections(state) {
      for (const button of elements.swatches) {
        button.classList.toggle('active', button.dataset.mode === state.paintMode);
      }

      for (const button of elements.toolButtons) {
        button.classList.toggle('active', button.dataset.tool === state.activeTool);
      }
    }

    /**
     * Mirrors opacity slider values into percentage labels.
     * @param {ViewState} view - Current view-only control values.
     * @returns {void}
     */
    function updateSliderLabels(view) {
      elements.opacityValue.textContent = `${Math.round(view.underlayOpacity * 100)}%`;
      elements.gridFillOpacityValue.textContent = `${Math.round(view.gridFillOpacity * 100)}%`;
      elements.gridOpacityValue.textContent = `${Math.round(view.gridOpacity * 100)}%`;
    }

    /**
     * Updates the live cursor readout.
     * @param {string} text - Readout text to display.
     * @returns {void}
     */
    function updateCursor(text) {
      elements.cursorReadout.textContent = text;
    }

    /**
     * Repaints the full on-screen editor preview.
     * @param {EditorState} state - Current editor state.
     * @param {ViewState} view - Current view-only control values.
     * @returns {void}
     */
    function renderCanvas(state, view) {
      // Draw layers in the same order the user perceives them.
      ctx.clearRect(0, 0, PREVIEW_W, PREVIEW_H);
      rendererDrawCheckerboard(ctx, PREVIEW_W, PREVIEW_H, 16);
      rendererDrawUnderlay(ctx, state.underlayImage, PREVIEW_W, PREVIEW_H, view.underlayOpacity);
      rendererDrawCells(ctx, state.cells, PREVIEW_CELL, view.gridFillOpacity);
      rendererDrawLinePreview(ctx, state, PREVIEW_CELL);
      rendererDrawGridLines(ctx, PREVIEW_W, PREVIEW_H, PREVIEW_CELL, view.gridOpacity);
    }

    return {
      refresh,
      renderExportCanvas,
      syncCanvasLayout,
      updateCursor,
    };
  }

  /**
   * Calculates a contain-style draw rectangle while preserving aspect ratio.
   * @param {HTMLImageElement | null} img - Source image to fit.
   * @param {number} width - Target draw area width.
   * @param {number} height - Target draw area height.
   * @returns {ContainRect | null}
   */
  function rendererGetContainRect(img, width, height) {
    if (!img) return null;

    const imageWidth = img.naturalWidth || img.width;
    const imageHeight = img.naturalHeight || img.height;
    if (!imageWidth || !imageHeight) return null;

    const imageRatio = imageWidth / imageHeight;
    const canvasRatio = width / height;
    if (imageRatio > canvasRatio) {
      const drawH = width / imageRatio;
      return { drawX: 0, drawY: (height - drawH) / 2, drawW: width, drawH };
    }

    const drawW = height * imageRatio;
    return { drawX: (width - drawW) / 2, drawY: 0, drawW, drawH: height };
  }

  /**
   * Draws the neutral checkerboard background for transparent cells.
   * @param {CanvasRenderingContext2D} context - Target 2D context.
   * @param {number} width - Draw area width.
   * @param {number} height - Draw area height.
   * @param {number} size - Square size for the checker pattern.
   * @returns {void}
   */
  function rendererDrawCheckerboard(context, width, height, size) {
    context.save();
    context.fillStyle = '#d8d8d8';
    context.fillRect(0, 0, width, height);
    context.fillStyle = '#c3c3c3';

    for (let y = 0; y < height; y += size) {
      for (let x = 0; x < width; x += size) {
        if (((x / size) + (y / size)) % 2 === 0) {
          context.fillRect(x, y, size, size);
        }
      }
    }

    context.restore();
  }

  /**
   * Draws the optional underlay image behind the grid.
   * @param {CanvasRenderingContext2D} context - Target 2D context.
   * @param {HTMLImageElement | null} underlayImage - Source underlay image, if any.
   * @param {number} width - Draw area width.
   * @param {number} height - Draw area height.
   * @param {number} opacity - Underlay alpha from `0` to `1`.
   * @returns {void}
   */
  function rendererDrawUnderlay(context, underlayImage, width, height, opacity) {
    if (!underlayImage || opacity <= 0) return;

    const rect = rendererGetContainRect(underlayImage, width, height);
    if (!rect) return;

    context.save();
    context.globalAlpha = opacity;
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(underlayImage, rect.drawX, rect.drawY, rect.drawW, rect.drawH);
    context.restore();
  }

  /**
   * Draws painted grid cells onto the target canvas.
   * @param {CanvasRenderingContext2D} context - Target 2D context.
   * @param {Uint8Array} cells - Flattened cell state array.
   * @param {number} cellSize - Draw size for one logical cell.
   * @param {number} opacity - Cell fill alpha from `0` to `1`.
   * @returns {void}
   */
  function rendererDrawCells(context, cells, cellSize, opacity) {
    context.save();
    context.globalAlpha = opacity;

    for (let row = 0; row < ROWS; row++) {
      for (let col = 0; col < COLS; col++) {
        const state = cells[(row * COLS) + col];
        if (state === STATE_TRANSPARENT) continue;
        context.fillStyle = state === STATE_WHITE ? '#ffffff' : '#000000';
        context.fillRect(col * cellSize, row * cellSize, cellSize, cellSize);
      }
    }

    context.restore();
  }

  /**
   * Draws the live line preview during an active line gesture.
   * @param {CanvasRenderingContext2D} context - Target 2D context.
   * @param {EditorState} state - Current editor state.
   * @param {number} cellSize - Draw size for one logical cell.
   * @returns {void}
   */
  function rendererDrawLinePreview(context, state, cellSize) {
    if (!state.isDrawingLine || !state.lineStartCell || !state.linePreviewCell) return;

    const target = getPaintTarget(state, state.cells[state.lineStartCell.idx], state.linePreviewShift);
    const previewColor = target === STATE_WHITE ? 'rgba(255, 255, 255, 0.72)' : 'rgba(0, 0, 0, 0.72)';

    context.save();

    for (const cell of getLineCells(state.lineStartCell, state.linePreviewCell)) {
      const x = cell.col * cellSize;
      const y = cell.row * cellSize;

      // Transparent preview uses a tinted overlay instead of literal transparency,
      // otherwise the preview would disappear into the checkerboard.
      if (target === STATE_TRANSPARENT) {
        context.fillStyle = 'rgba(255, 119, 119, 0.28)';
        context.fillRect(x, y, cellSize, cellSize);
        context.strokeStyle = 'rgba(255, 119, 119, 0.8)';
        context.strokeRect(x + 1, y + 1, Math.max(1, cellSize - 2), Math.max(1, cellSize - 2));
        continue;
      }

      context.fillStyle = previewColor;
      context.fillRect(x, y, cellSize, cellSize);
    }

    context.restore();
  }

  /**
   * Draws the grid-line overlay on top of the preview.
   * @param {CanvasRenderingContext2D} context - Target 2D context.
   * @param {number} width - Draw area width.
   * @param {number} height - Draw area height.
   * @param {number} cellSize - Draw size for one logical cell.
   * @param {number} opacity - Grid line alpha from `0` to `1`.
   * @returns {void}
   */
  function rendererDrawGridLines(context, width, height, cellSize, opacity) {
    if (opacity <= 0) return;

    context.save();
    context.strokeStyle = `rgba(116, 167, 255, ${opacity})`;
    context.lineWidth = 1;
    context.beginPath();

    for (let x = 0; x <= width; x += cellSize) {
      context.moveTo(x + 0.5, 0);
      context.lineTo(x + 0.5, height);
    }

    for (let y = 0; y <= height; y += cellSize) {
      context.moveTo(0, y + 0.5);
      context.lineTo(width, y + 0.5);
    }

    context.stroke();
    context.restore();
  }

  EditorApp.createEditorRenderer = createEditorRenderer;
}
