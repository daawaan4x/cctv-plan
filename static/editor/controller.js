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
    STATE_TRANSPARENT,
    applyBrushAtCell,
    applyFillAtCell,
    applyImportedCells,
    beginBrushStroke,
    cancelLine,
    clearGrid,
    clearImportedGridLabel,
    clearUnderlay,
    createEditorRenderer,
    createEditorState,
    downloadCanvasAsPng,
    endBrushStroke,
    finishLine,
    fitZoom,
    importGridCellsFromImage,
    loadImageViaFileReader,
    redo,
    setActiveTool,
    setGridImportStatusMessage,
    setPaintMode,
    setUnderlay,
    setUnderlayStatusMessage,
    startLine,
    undo,
    updateLinePreview,
    zoomIn,
    zoomOut,
  } = EditorApp;

  /**
   * Optional flags that affect one refresh pass.
   * @typedef {{ layout?: boolean }} RefreshOptions
   */

  /**
   * State transition used by undo and redo controls.
   * @typedef {(state: EditorState) => boolean} HistoryAction
   */

  /**
   * State transition used by zoom controls.
   * @typedef {(state: EditorState) => void} ZoomAction
   */

  /**
   * Connects the editor model, renderer, and DOM event handlers.
   * @param {Document} [root=document] - Document containing the editor markup.
   * @returns {EditorController}
   */
  function createEditorController(root = document) {
    const elements = controllerGetElements(root);
    const state = createEditorState();
    const renderer = createEditorRenderer(elements);

    let cursorText = 'Cell: —';
    /** @type {number | null} */
    let activePointerId = null;

    bindEvents();
    refresh({ layout: true });

    return { state };

    // Event wiring

    /**
     * Binds DOM events for pointer input, toolbar actions, and shortcuts.
     * @returns {void}
     */
    function bindEvents() {
      elements.canvas.addEventListener('pointerdown', handlePointerDown);
      elements.canvas.addEventListener('pointermove', handlePointerMove);
      elements.canvas.addEventListener('pointerup', handlePointerUp);
      elements.canvas.addEventListener('pointercancel', handlePointerCancel);
      elements.canvas.addEventListener('pointerleave', () => updateCursor('Cell: —'));

      for (const button of elements.swatches) {
        button.addEventListener('click', () => {
          const paintMode = button.dataset.mode;
          if (!paintMode) return;

          setPaintMode(state, /** @type {PaintMode} */ (paintMode));
          refresh();
        });
      }

      for (const button of elements.toolButtons) {
        button.addEventListener('click', () => {
          const activeTool = button.dataset.tool;
          if (!activeTool) return;

          if (state.isDrawingLine && activeTool !== 'line') {
            cancelLine(state);
            releasePointerCapture();
          }

          setActiveTool(state, /** @type {ActiveTool} */ (activeTool));
          refresh();
        });
      }

      elements.imageInput.addEventListener('change', handleUnderlayImport);
      elements.gridImageInput.addEventListener('change', handleGridImport);
      elements.removeUnderlayBtn.addEventListener('click', handleRemoveUnderlay);

      elements.zoomOutBtn.addEventListener('click', () => applyZoom(zoomOut));
      elements.zoomFitBtn.addEventListener('click', () => applyZoom(fitZoom));
      elements.zoomInBtn.addEventListener('click', () => applyZoom(zoomIn));

      elements.underlayOpacity.addEventListener('input', () => refresh());
      elements.gridFillOpacity.addEventListener('input', () => refresh());
      elements.gridOpacity.addEventListener('input', () => refresh());

      elements.clearBtn.addEventListener('click', handleClearGrid);
      elements.undoBtn.addEventListener('click', () => applyHistoryAction(undo));
      elements.redoBtn.addEventListener('click', () => applyHistoryAction(redo));
      elements.exportBtn.addEventListener('click', handleExport);

      window.addEventListener('keydown', handleKeydown);
      window.addEventListener('resize', () => renderer.syncCanvasLayout(state.zoomScale));
    }

    // View synchronization

    /**
     * Reads the current view-only control values from the DOM.
     * @returns {ViewState}
     */
    function getViewState() {
      return {
        underlayOpacity: Number(elements.underlayOpacity.value) / 100,
        gridFillOpacity: Number(elements.gridFillOpacity.value) / 100,
        gridOpacity: Number(elements.gridOpacity.value) / 100,
      };
    }

    /**
     * Re-renders the editor and optionally recalculates layout sizing first.
     * @param {RefreshOptions} [options={}] - Optional render flags.
     * @returns {void}
     */
    function refresh(options = {}) {
      if (options.layout) {
        renderer.syncCanvasLayout(state.zoomScale);
      }

      renderer.refresh(state, getViewState(), cursorText);
      elements.canvas.style.cursor = state.activeTool === 'fill' ? 'cell' : 'crosshair';
    }

    /**
     * Updates the current cursor readout text.
     * @param {string} nextText - Readout text to display.
     * @returns {void}
     */
    function updateCursor(nextText) {
      cursorText = nextText;
      renderer.updateCursor(cursorText);
    }

    /**
     * Formats the standard cell readout for the cursor status.
     * @param {Cell} cell - Logical grid cell under the pointer.
     * @returns {string}
     */
    function formatCellText(cell) {
      return `Cell: column ${cell.col + 1}, row ${cell.row + 1}`;
    }

    /**
     * Formats line-tool start or end text for the cursor status.
     * @param {string} label - Prefix describing the point role.
     * @param {Cell} cell - Logical grid cell under the pointer.
     * @returns {string}
     */
    function formatLineText(label, cell) {
      return `${label}: column ${cell.col + 1}, row ${cell.row + 1}`;
    }

    // Pointer helpers

    /**
     * Converts a canvas pointer event into a logical grid cell.
     * @param {PointerEvent} event - Pointer event relative to the canvas.
     * @returns {Cell | null}
     */
    function pointToCell(event) {
      const rect = elements.canvas.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width;
      const y = (event.clientY - rect.top) / rect.height;
      const col = Math.floor(x * COLS);
      const row = Math.floor(y * ROWS);

      // Normalize against the displayed canvas size so zoom never changes
      // which logical cell the pointer lands on.
      if (col < 0 || col >= COLS || row < 0 || row >= ROWS) return null;
      return { col, row, idx: (row * COLS) + col };
    }

    /**
     * Captures the active pointer so drags continue outside the canvas bounds.
     * @param {PointerEvent} event - Pointer event that began the gesture.
     * @returns {void}
     */
    function capturePointer(event) {
      activePointerId = event.pointerId;
      elements.canvas.setPointerCapture(activePointerId);
    }

    /**
     * Releases pointer capture when a gesture ends or is cancelled.
     * @returns {void}
     */
    function releasePointerCapture() {
      if (activePointerId === null) return;

      try {
        elements.canvas.releasePointerCapture(activePointerId);
      } catch (_) {}

      activePointerId = null;
    }

    // Pointer interactions

    /**
     * Starts the gesture for the active tool.
     * @param {PointerEvent} event - Pointer-down event on the canvas.
     * @returns {void}
     */
    function handlePointerDown(event) {
      event.preventDefault();
      const cell = pointToCell(event);
      if (!cell) return;

      if (state.activeTool === 'fill') {
        updateCursor(formatCellText(cell));
        if (applyFillAtCell(state, cell, event.shiftKey)) {
          refresh();
        }
        return;
      }

      if (state.activeTool === 'line') {
        // Line mode enters a preview gesture and waits until pointer-up to commit.
        if (!startLine(state, cell, event.shiftKey)) return;
        capturePointer(event);
        cursorText = formatLineText('Line start', cell);
        refresh();
        return;
      }

      beginBrushStroke(state);
      capturePointer(event);
      cursorText = formatCellText(cell);

      if (applyBrushAtCell(state, cell, event.shiftKey)) {
        refresh();
        return;
      }

      renderer.updateCursor(cursorText);
    }

    /**
     * Updates hover state or the currently active gesture during pointer movement.
     * @param {PointerEvent} event - Pointer-move event on the canvas.
     * @returns {void}
     */
    function handlePointerMove(event) {
      const cell = pointToCell(event);
      if (!cell) {
        updateCursor('Cell: —');
        return;
      }

      if (state.isDrawingLine) {
        // The line preview follows the pointer continuously, but the grid
        // itself stays unchanged until the gesture is finished.
        if (!updateLinePreview(state, cell, event.shiftKey)) return;
        cursorText = formatLineText('Line end', cell);
        refresh();
        return;
      }

      cursorText = formatCellText(cell);
      if (state.activeTool === 'brush' && state.isPainting && applyBrushAtCell(state, cell, event.shiftKey)) {
        refresh();
        return;
      }

      renderer.updateCursor(cursorText);
    }

    /**
     * Finishes the active pointer gesture.
     * @param {PointerEvent} event - Pointer-up event on the canvas.
     * @returns {void}
     */
    function handlePointerUp(event) {
      if (state.isDrawingLine) {
        finishLine(state, pointToCell(event) || state.linePreviewCell, event.shiftKey);
        releasePointerCapture();
        refresh();
        return;
      }

      endBrushStroke(state);
      releasePointerCapture();
      refresh();
    }

    /**
     * Aborts the active pointer gesture when the browser cancels it.
     * @returns {void}
     */
    function handlePointerCancel() {
      if (state.isDrawingLine) {
        cancelLine(state);
        releasePointerCapture();
        refresh();
        return;
      }

      endBrushStroke(state);
      releasePointerCapture();
      refresh();
    }

    // File actions

    /**
     * Loads the selected underlay image file.
     * @returns {Promise<void>}
     */
    async function handleUnderlayImport() {
      const file = elements.imageInput.files && elements.imageInput.files[0];
      if (!file) return;

      setUnderlayStatusMessage(state, `Underlay: loading ${file.name}…`);
      refresh();

      try {
        const image = await loadImageViaFileReader(file);
        setUnderlay(state, image, file.name);
      } catch (error) {
        clearUnderlay(state);
        setUnderlayStatusMessage(state, `Underlay: failed to load ${file.name}`);
        alert(controllerGetErrorMessage(error, 'The selected file could not be loaded as an image.'));
      }

      refresh();
    }

    /**
     * Loads and applies the selected grid image file.
     * @returns {Promise<void>}
     */
    async function handleGridImport() {
      const file = elements.gridImageInput.files && elements.gridImageInput.files[0];
      if (!file) return;

      setGridImportStatusMessage(state, `Grid import: loading ${file.name}…`);
      refresh();

      try {
        const importedImage = await loadImageViaFileReader(file);
        const imported = importGridCellsFromImage(importedImage, file.name);
        applyImportedCells(
          state,
          imported.cells,
          `Grid import: ${file.name} (${imported.width}×${imported.height}, scale ×${imported.scale})`,
        );
      } catch (error) {
        clearImportedGridLabel(state);
        setGridImportStatusMessage(state, `Grid import: failed to load ${file.name}`);
        alert(controllerGetErrorMessage(error, 'The selected grid image could not be imported.'));
      } finally {
        elements.gridImageInput.value = '';
      }

      refresh();
    }

    /**
     * Removes the current underlay image.
     * @returns {void}
     */
    function handleRemoveUnderlay() {
      clearUnderlay(state);
      elements.imageInput.value = '';
      refresh();
    }

    /**
     * Clears the full grid after user confirmation.
     * @returns {void}
     */
    function handleClearGrid() {
      const hasPaintedCells = state.cells.some(
        /** @param {number} value */
        (value) => value !== STATE_TRANSPARENT,
      );
      if (!hasPaintedCells) return;
      if (!confirm('Clear all grid cells to transparent?')) return;

      if (clearGrid(state)) {
        refresh();
      }
    }

    /**
     * Runs an undo/redo-style state transition and refreshes if it changed state.
     * @param {HistoryAction} action - History action to execute.
     * @returns {void}
     */
    function applyHistoryAction(action) {
      if (!action(state)) return;
      refresh();
    }

    /**
     * Runs a zoom action and resynchronizes layout sizing.
     * @param {ZoomAction} action - Zoom action to execute.
     * @returns {void}
     */
    function applyZoom(action) {
      action(state);
      refresh({ layout: true });
    }

    /**
     * Builds the export canvas and starts a download.
     * @returns {void}
     */
    function handleExport() {
      const canvasToDownload = renderer.renderExportCanvas(state, {
        scale: Number(elements.exportScale.value),
        includeUnderlay: elements.includeUnderlay.value,
        underlayOpacity: Number(elements.underlayOpacity.value) / 100,
      });

      downloadCanvasAsPng(canvasToDownload);
    }

    /**
     * Handles editor keyboard shortcuts.
     * @param {KeyboardEvent} event - Key event from the window.
     * @returns {void}
     */
    function handleKeydown(event) {
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      const key = event.key.toLowerCase();
      const mod = event.ctrlKey || event.metaKey;

      if (mod && key === 's') {
        event.preventDefault();
        handleExport();
        return;
      }
      if (key === '=' || key === '+') {
        event.preventDefault();
        applyZoom(zoomIn);
        return;
      }
      if (key === '-') {
        event.preventDefault();
        applyZoom(zoomOut);
        return;
      }
      if (key === '0') {
        event.preventDefault();
        applyZoom(fitZoom);
        return;
      }
      if (mod && key === 'z' && event.shiftKey) {
        event.preventDefault();
        applyHistoryAction(redo);
        return;
      }
      if (mod && key === 'z') {
        event.preventDefault();
        applyHistoryAction(undo);
        return;
      }
      if (mod && key === 'y') {
        event.preventDefault();
        applyHistoryAction(redo);
        return;
      }

      if (event.key === '1') controllerClickButtonByDataset(elements.swatches, 'mode', 'cycle');
      if (event.key === '2') controllerClickButtonByDataset(elements.swatches, 'mode', 'transparent');
      if (event.key === '3') controllerClickButtonByDataset(elements.swatches, 'mode', 'white');
      if (event.key === '4') controllerClickButtonByDataset(elements.swatches, 'mode', 'black');
      if (key === 'b') controllerClickButtonByDataset(elements.toolButtons, 'tool', 'brush');
      if (key === 'f') controllerClickButtonByDataset(elements.toolButtons, 'tool', 'fill');
      if (key === 'l') controllerClickButtonByDataset(elements.toolButtons, 'tool', 'line');
    }
  }

  /**
   * Normalizes thrown values into user-facing error text.
   * @param {unknown} error - Unknown thrown value.
   * @param {string} fallback - Fallback message when the error is not an `Error`.
   * @returns {string}
   */
  function controllerGetErrorMessage(error, fallback) {
    return error instanceof Error ? error.message : fallback;
  }

  /**
   * Finds a toolbar button by dataset value and activates it.
   * @param {HTMLButtonElement[]} buttons - Candidate toolbar buttons.
   * @param {'mode' | 'tool'} key - Dataset key to match.
   * @param {string} value - Dataset value to match.
   * @returns {void}
   */
  function controllerClickButtonByDataset(buttons, key, value) {
    const button = buttons.find((candidate) => candidate.dataset[key] === value);
    if (button) button.click();
  }

  /**
   * Collects the DOM elements required by the editor runtime.
   * @param {Document} root - Document containing the editor markup.
   * @returns {EditorElements}
   */
  function controllerGetElements(root) {
    return {
      canvas: /** @type {HTMLCanvasElement} */ (controllerRequireById(root, 'editor')),
      canvasScroll: /** @type {HTMLElement} */ (controllerRequireQuery(root, '.canvas-scroll')),
      canvasShell: /** @type {HTMLElement} */ (controllerRequireById(root, 'canvasShell')),
      zoomOutBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'zoomOutBtn')),
      zoomFitBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'zoomFitBtn')),
      zoomInBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'zoomInBtn')),
      zoomValue: /** @type {HTMLElement} */ (controllerRequireById(root, 'zoomValue')),
      imageInput: /** @type {HTMLInputElement} */ (controllerRequireById(root, 'imageInput')),
      gridImageInput: /** @type {HTMLInputElement} */ (controllerRequireById(root, 'gridImageInput')),
      removeUnderlayBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'removeUnderlayBtn')),
      underlayOpacity: /** @type {HTMLInputElement} */ (controllerRequireById(root, 'underlayOpacity')),
      opacityValue: /** @type {HTMLElement} */ (controllerRequireById(root, 'opacityValue')),
      gridFillOpacity: /** @type {HTMLInputElement} */ (controllerRequireById(root, 'gridFillOpacity')),
      gridFillOpacityValue: /** @type {HTMLElement} */ (controllerRequireById(root, 'gridFillOpacityValue')),
      gridOpacity: /** @type {HTMLInputElement} */ (controllerRequireById(root, 'gridOpacity')),
      gridOpacityValue: /** @type {HTMLElement} */ (controllerRequireById(root, 'gridOpacityValue')),
      exportScale: /** @type {HTMLSelectElement} */ (controllerRequireById(root, 'exportScale')),
      includeUnderlay: /** @type {HTMLSelectElement} */ (controllerRequireById(root, 'includeUnderlay')),
      clearBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'clearBtn')),
      exportBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'exportBtn')),
      undoBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'undoBtn')),
      redoBtn: /** @type {HTMLButtonElement} */ (controllerRequireById(root, 'redoBtn')),
      status: /** @type {HTMLElement} */ (controllerRequireById(root, 'status')),
      underlayStatus: /** @type {HTMLElement} */ (controllerRequireById(root, 'underlayStatus')),
      gridImportStatus: /** @type {HTMLElement} */ (controllerRequireById(root, 'gridImportStatus')),
      cursorReadout: /** @type {HTMLElement} */ (controllerRequireById(root, 'cursorReadout')),
      swatches: [...root.querySelectorAll('.swatch')].map((element) => /** @type {HTMLButtonElement} */ (element)),
      toolButtons: [...root.querySelectorAll('.segment')].map((element) => /** @type {HTMLButtonElement} */ (element)),
    };
  }

  /**
   * Reads a required DOM element by id.
   * @template {HTMLElement} T
   * @param {Document} root - Document containing the editor markup.
   * @param {string} id - Element id to resolve.
   * @returns {T}
   */
  function controllerRequireById(root, id) {
    const element = root.getElementById(id);
    if (!element) {
      throw new Error(`Required editor element #${id} was not found.`);
    }

    return /** @type {T} */ (element);
  }

  /**
   * Reads a required DOM element by selector.
   * @template {HTMLElement} T
   * @param {Document} root - Document containing the editor markup.
   * @param {string} selector - Selector to resolve.
   * @returns {T}
   */
  function controllerRequireQuery(root, selector) {
    const element = root.querySelector(selector);
    if (!element) {
      throw new Error(`Required editor element ${selector} was not found.`);
    }

    return /** @type {T} */ (element);
  }

  EditorApp.createEditorController = createEditorController;
}
