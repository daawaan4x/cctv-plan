/// <reference path="./types.js" />
// @ts-check

{
  /** @type {EditorWindow} */
  const editorWindow = /** @type {EditorWindow} */ (window);
  editorWindow.EditorApp = /** @type {EditorAppNamespace} */ (editorWindow.EditorApp || {});
  const EditorApp = editorWindow.EditorApp;
  const {
    COLS,
    MAX_HISTORY,
    MAX_ZOOM,
    MIN_ZOOM,
    ROWS,
    STATE_BLACK,
    STATE_TRANSPARENT,
    STATE_WHITE,
    ZOOM_STEP,
  } = EditorApp;

  // State creation

  /**
   * Creates the mutable editor state container used by the controller and renderer.
   * @returns {EditorState}
   */
  function createEditorState() {
    return {
      cells: new Uint8Array(COLS * ROWS),
      underlayImage: null,
      underlayName: '',
      underlayStatusMessage: '',
      importedGridName: '',
      gridImportStatusMessage: '',
      paintMode: 'cycle',
      activeTool: 'brush',
      isPainting: false,
      isDrawingLine: false,
      lineStartCell: null,
      linePreviewCell: null,
      linePreviewShift: false,
      gestureChanged: false,
      historyBeforeGesture: null,
      lastPaintedIndex: -1,
      undoStack: [],
      redoStack: [],
      zoomScale: MIN_ZOOM,
    };
  }

  /**
   * Converts a grid column and row into a flat array index.
   * @param {number} col - Zero-based grid column.
   * @param {number} row - Zero-based grid row.
   * @returns {number}
   */
  function indexOf(col, row) {
    return row * COLS + col;
  }

  /**
   * Clamps a numeric value into the provided bounds.
   * @param {number} value - Value to constrain.
   * @param {number} min - Inclusive lower bound.
   * @param {number} max - Inclusive upper bound.
   * @returns {number}
   */
  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  /**
   * Updates the selected paint mode.
   * @param {EditorState} state - Mutable editor state.
   * @param {PaintMode} paintMode - Toolbar paint mode to activate.
   * @returns {void}
   */
  function setPaintMode(state, paintMode) {
    state.paintMode = paintMode;
  }

  /**
   * Updates the selected drawing tool.
   * @param {EditorState} state - Mutable editor state.
   * @param {ActiveTool} activeTool - Tool to activate.
   * @returns {void}
   */
  function setActiveTool(state, activeTool) {
    state.activeTool = activeTool;
  }

  /**
   * Counts the current cell states for the status readout.
   * @param {EditorState} state - Mutable editor state.
   * @returns {CellCounts}
   */
  function countCells(state) {
    let transparent = 0;
    let white = 0;
    let black = 0;

    for (const value of state.cells) {
      if (value === STATE_WHITE) {
        white++;
        continue;
      }
      if (value === STATE_BLACK) {
        black++;
        continue;
      }
      transparent++;
    }

    return { transparent, white, black };
  }

  // Zoom

  /**
   * Applies a zoom value after clamping it to editor limits.
   * @param {EditorState} state - Mutable editor state.
   * @param {number} nextZoom - Requested zoom multiplier.
   * @returns {void}
   */
  function setZoom(state, nextZoom) {
    state.zoomScale = clamp(nextZoom, MIN_ZOOM, MAX_ZOOM);
  }

  /**
   * Increases the editor zoom by one configured step.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function zoomIn(state) {
    setZoom(state, state.zoomScale * ZOOM_STEP);
  }

  /**
   * Decreases the editor zoom by one configured step.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function zoomOut(state) {
    setZoom(state, state.zoomScale / ZOOM_STEP);
  }

  /**
   * Resets the zoom back to the fitted baseline scale.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function fitZoom(state) {
    setZoom(state, MIN_ZOOM);
  }

  // Undo / redo

  /**
   * Pushes a snapshot onto the undo stack and clears redo history.
   * @param {EditorState} state - Mutable editor state.
   * @param {Uint8Array | null} snapshot - Previous grid snapshot to retain.
   * @returns {void}
   */
  function pushUndoSnapshot(state, snapshot) {
    if (!snapshot) return;

    state.undoStack.push(snapshot);
    if (state.undoStack.length > MAX_HISTORY) {
      state.undoStack.shift();
    }
    state.redoStack = [];
  }

  /**
   * Captures the current grid as an undo checkpoint.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function pushCurrentToUndo(state) {
    pushUndoSnapshot(state, state.cells.slice());
  }

  /**
   * Restores the most recent undo snapshot.
   * @param {EditorState} state - Mutable editor state.
   * @returns {boolean}
   */
  function undo(state) {
    if (!state.undoStack.length) return false;

    const snapshot = state.undoStack.pop();
    if (!snapshot) return false;

    state.redoStack.push(state.cells.slice());
    state.cells.set(snapshot);
    return true;
  }

  /**
   * Restores the most recent redo snapshot.
   * @param {EditorState} state - Mutable editor state.
   * @returns {boolean}
   */
  function redo(state) {
    if (!state.redoStack.length) return false;

    const snapshot = state.redoStack.pop();
    if (!snapshot) return false;

    state.undoStack.push(state.cells.slice());
    state.cells.set(snapshot);
    return true;
  }

  // Underlay and import status

  /**
   * Stores the currently loaded underlay image metadata.
   * @param {EditorState} state - Mutable editor state.
   * @param {HTMLImageElement} image - Decoded underlay image.
   * @param {string} name - Original filename or display label.
   * @returns {void}
   */
  function setUnderlay(state, image, name) {
    state.underlayImage = image;
    state.underlayName = name;
    state.underlayStatusMessage = '';
  }

  /**
   * Removes the active underlay image and its label.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function clearUnderlay(state) {
    state.underlayImage = null;
    state.underlayName = '';
    state.underlayStatusMessage = '';
  }

  /**
   * Overrides the default underlay status text.
   * @param {EditorState} state - Mutable editor state.
   * @param {string} message - Status message to display.
   * @returns {void}
   */
  function setUnderlayStatusMessage(state, message) {
    state.underlayStatusMessage = message;
  }

  /**
   * Computes the underlay status line shown in the UI.
   * @param {EditorState} state - Mutable editor state.
   * @returns {string}
   */
  function getUnderlayStatus(state) {
    if (state.underlayStatusMessage) return state.underlayStatusMessage;
    if (!state.underlayImage) return 'Underlay: none';
    return `Underlay: ${state.underlayName || 'loaded image'} (${state.underlayImage.width}×${state.underlayImage.height})`;
  }

  /**
   * Clears the imported-grid status label.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function clearImportedGridLabel(state) {
    state.importedGridName = '';
  }

  /**
   * Stores the imported-grid status label.
   * @param {EditorState} state - Mutable editor state.
   * @param {string} label - Status label describing the imported grid.
   * @returns {void}
   */
  function setImportedGridLabel(state, label) {
    state.importedGridName = label;
    state.gridImportStatusMessage = '';
  }

  /**
   * Overrides the default grid import status text.
   * @param {EditorState} state - Mutable editor state.
   * @param {string} message - Status message to display.
   * @returns {void}
   */
  function setGridImportStatusMessage(state, message) {
    state.gridImportStatusMessage = message;
  }

  /**
   * Computes the grid import status line shown in the UI.
   * @param {EditorState} state - Mutable editor state.
   * @returns {string}
   */
  function getGridImportStatus(state) {
    if (state.gridImportStatusMessage) return state.gridImportStatusMessage;
    if (!state.importedGridName) return 'Grid import: none';
    return state.importedGridName;
  }

  // Paint tools

  /**
   * Returns the next value in cycle paint mode.
   * @param {number} current - Current stored cell state.
   * @returns {number}
   */
  function nextState(current) {
    if (current === STATE_TRANSPARENT) return STATE_WHITE;
    if (current === STATE_WHITE) return STATE_BLACK;
    return STATE_TRANSPARENT;
  }

  /**
   * Resolves which state should be written for the current gesture.
   * @param {EditorState} state - Mutable editor state.
   * @param {number} current - Current stored cell state.
   * @param {boolean} [shiftKey=false] - Whether erase-modifier behavior is active.
   * @returns {number}
   */
  function getPaintTarget(state, current, shiftKey = false) {
    // Shift always acts as a temporary erase shortcut, regardless of the
    // currently selected paint mode.
    if (shiftKey) return STATE_TRANSPARENT;
    if (state.paintMode === 'transparent') return STATE_TRANSPARENT;
    if (state.paintMode === 'white') return STATE_WHITE;
    if (state.paintMode === 'black') return STATE_BLACK;
    return nextState(current);
  }

  /**
   * Starts a brush gesture and captures the pre-gesture snapshot for undo.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function beginBrushStroke(state) {
    state.isPainting = true;
    state.gestureChanged = false;
    state.historyBeforeGesture = state.cells.slice();
    state.lastPaintedIndex = -1;
  }

  /**
   * Applies the brush to one grid cell during an active drag gesture.
   * @param {EditorState} state - Mutable editor state.
   * @param {Cell | null} cell - Cell under the pointer, if any.
   * @param {boolean} shiftKey - Whether erase-modifier behavior is active.
   * @returns {boolean}
   */
  function applyBrushAtCell(state, cell, shiftKey) {
    if (!state.isPainting || !cell || cell.idx === state.lastPaintedIndex) return false;

    const target = getPaintTarget(state, state.cells[cell.idx], shiftKey);
    const changed = state.cells[cell.idx] !== target;

    if (changed) {
      state.cells[cell.idx] = target;
      state.gestureChanged = true;
    }

    state.lastPaintedIndex = cell.idx;
    return changed;
  }

  /**
   * Finalizes the active brush gesture and records undo history when needed.
   * @param {EditorState} state - Mutable editor state.
   * @returns {boolean}
   */
  function endBrushStroke(state) {
    const changed = state.isPainting && state.gestureChanged;

    if (changed) {
      pushUndoSnapshot(state, state.historyBeforeGesture);
    }

    state.isPainting = false;
    state.gestureChanged = false;
    state.historyBeforeGesture = null;
    state.lastPaintedIndex = -1;
    return changed;
  }

  /**
   * Recolors one connected region in the flattened grid.
   * @param {Uint8Array} cells - Grid cells to mutate in place.
   * @param {number} startIdx - Starting cell index.
   * @param {number} targetState - State to write across the connected region.
   * @returns {number}
   */
  function bucketFill(cells, startIdx, targetState) {
    const sourceState = cells[startIdx];
    if (sourceState === targetState) return 0;

    let changed = 0;
    const stack = [startIdx];
    const seen = new Uint8Array(cells.length);
    seen[startIdx] = 1;

    while (stack.length) {
      const idx = stack.pop();
      if (idx === undefined) continue;
      if (cells[idx] !== sourceState) continue;

      cells[idx] = targetState;
      changed++;

      // Walk neighbors in the flattened array instead of recursing so large fills
      // cannot overflow the browser call stack.
      const col = idx % COLS;
      const row = Math.floor(idx / COLS);
      const left = idx - 1;
      const right = idx + 1;
      const up = idx - COLS;
      const down = idx + COLS;

      if (col > 0 && !seen[left]) {
        seen[left] = 1;
        stack.push(left);
      }
      if (col < COLS - 1 && !seen[right]) {
        seen[right] = 1;
        stack.push(right);
      }
      if (row > 0 && !seen[up]) {
        seen[up] = 1;
        stack.push(up);
      }
      if (row < ROWS - 1 && !seen[down]) {
        seen[down] = 1;
        stack.push(down);
      }
    }

    return changed;
  }

  /**
   * Flood-fills the clicked region with the resolved paint target.
   * @param {EditorState} state - Mutable editor state.
   * @param {Cell | null} cell - Seed cell for the fill.
   * @param {boolean} shiftKey - Whether erase-modifier behavior is active.
   * @returns {boolean}
   */
  function applyFillAtCell(state, cell, shiftKey) {
    if (!cell) return false;

    const target = getPaintTarget(state, state.cells[cell.idx], shiftKey);
    const before = state.cells.slice();
    const changed = bucketFill(state.cells, cell.idx, target);

    if (changed) {
      pushUndoSnapshot(state, before);
    }

    return changed > 0;
  }

  // Line tool

  /**
   * Returns all grid cells crossed by a line between two endpoints.
   * @param {Cell} start - Line start cell.
   * @param {Cell} end - Line end cell.
   * @returns {Cell[]}
   */
  function getLineCells(start, end) {
    /** @type {Cell[]} */
    const result = [];
    let x0 = start.col;
    let y0 = start.row;
    const x1 = end.col;
    const y1 = end.row;
    const dx = Math.abs(x1 - x0);
    const sx = x0 < x1 ? 1 : -1;
    const dy = -Math.abs(y1 - y0);
    const sy = y0 < y1 ? 1 : -1;
    let error = dx + dy;

    while (true) {
      result.push({ col: x0, row: y0, idx: indexOf(x0, y0) });
      if (x0 === x1 && y0 === y1) break;

      // Match classic Bresenham stepping so line preview and committed output
      // follow the same rasterized path through the logical grid.
      const doubledError = 2 * error;
      if (doubledError >= dy) {
        error += dy;
        x0 += sx;
      }
      if (doubledError <= dx) {
        error += dx;
        y0 += sy;
      }
    }

    return result;
  }

  /**
   * Starts a line-drawing gesture from the given cell.
   * @param {EditorState} state - Mutable editor state.
   * @param {Cell | null} cell - Starting cell under the pointer.
   * @param {boolean} shiftKey - Whether erase-modifier behavior is active.
   * @returns {boolean}
   */
  function startLine(state, cell, shiftKey) {
    if (!cell) return false;

    state.isDrawingLine = true;
    state.lineStartCell = cell;
    state.linePreviewCell = cell;
    state.linePreviewShift = shiftKey;
    state.historyBeforeGesture = state.cells.slice();
    return true;
  }

  /**
   * Updates the live line preview endpoint.
   * @param {EditorState} state - Mutable editor state.
   * @param {Cell | null} cell - Current cell under the pointer.
   * @param {boolean} shiftKey - Whether erase-modifier behavior is active.
   * @returns {boolean}
   */
  function updateLinePreview(state, cell, shiftKey) {
    if (!state.isDrawingLine || !state.lineStartCell || !cell) return false;

    state.linePreviewCell = cell;
    state.linePreviewShift = shiftKey;
    return true;
  }

  /**
   * Commits the previewed line into the grid.
   * @param {EditorState} state - Mutable editor state.
   * @param {Cell | null} endCell - Final endpoint if one is available.
   * @param {boolean} shiftKey - Whether erase-modifier behavior is active.
   * @returns {boolean}
   */
  function finishLine(state, endCell, shiftKey) {
    if (!state.isDrawingLine || !state.lineStartCell || !state.linePreviewCell) {
      cancelLine(state);
      return false;
    }

    const targetCell = endCell || state.linePreviewCell;
    const target = getPaintTarget(state, state.cells[state.lineStartCell.idx], shiftKey);
    let changed = 0;

    for (const cell of getLineCells(state.lineStartCell, targetCell)) {
      if (state.cells[cell.idx] === target) continue;
      state.cells[cell.idx] = target;
      changed++;
    }

    if (changed) {
      pushUndoSnapshot(state, state.historyBeforeGesture || state.cells.slice());
    }

    cancelLine(state);
    return changed > 0;
  }

  /**
   * Cancels the active line gesture and clears preview state.
   * @param {EditorState} state - Mutable editor state.
   * @returns {void}
   */
  function cancelLine(state) {
    state.isDrawingLine = false;
    state.lineStartCell = null;
    state.linePreviewCell = null;
    state.linePreviewShift = false;
    state.historyBeforeGesture = null;
  }

  // Import / clear helpers

  /**
   * Replaces the grid with imported cell values.
   * @param {EditorState} state - Mutable editor state.
   * @param {Uint8Array} cells - Imported cell payload.
   * @param {string} label - Status label describing the import.
   * @returns {boolean}
   */
  function applyImportedCells(state, cells, label) {
    const changed = state.cells.some((value, idx) => value !== cells[idx]);

    if (changed) {
      pushCurrentToUndo(state);
      state.cells.set(cells);
    }

    setImportedGridLabel(state, label);
    return changed;
  }

  /**
   * Clears the entire grid back to transparent.
   * @param {EditorState} state - Mutable editor state.
   * @returns {boolean}
   */
  function clearGrid(state) {
    const hasPaintedCells = state.cells.some((value) => value !== STATE_TRANSPARENT);
    if (!hasPaintedCells) return false;

    pushCurrentToUndo(state);
    state.cells.fill(STATE_TRANSPARENT);
    return true;
  }

  EditorApp.createEditorState = createEditorState;
  EditorApp.indexOf = indexOf;
  EditorApp.clamp = clamp;
  EditorApp.setPaintMode = setPaintMode;
  EditorApp.setActiveTool = setActiveTool;
  EditorApp.countCells = countCells;
  EditorApp.setZoom = setZoom;
  EditorApp.zoomIn = zoomIn;
  EditorApp.zoomOut = zoomOut;
  EditorApp.fitZoom = fitZoom;
  EditorApp.undo = undo;
  EditorApp.redo = redo;
  EditorApp.setUnderlay = setUnderlay;
  EditorApp.clearUnderlay = clearUnderlay;
  EditorApp.setUnderlayStatusMessage = setUnderlayStatusMessage;
  EditorApp.getUnderlayStatus = getUnderlayStatus;
  EditorApp.clearImportedGridLabel = clearImportedGridLabel;
  EditorApp.setImportedGridLabel = setImportedGridLabel;
  EditorApp.setGridImportStatusMessage = setGridImportStatusMessage;
  EditorApp.getGridImportStatus = getGridImportStatus;
  EditorApp.nextState = nextState;
  EditorApp.getPaintTarget = getPaintTarget;
  EditorApp.beginBrushStroke = beginBrushStroke;
  EditorApp.applyBrushAtCell = applyBrushAtCell;
  EditorApp.endBrushStroke = endBrushStroke;
  EditorApp.applyFillAtCell = applyFillAtCell;
  EditorApp.getLineCells = getLineCells;
  EditorApp.startLine = startLine;
  EditorApp.updateLinePreview = updateLinePreview;
  EditorApp.finishLine = finishLine;
  EditorApp.cancelLine = cancelLine;
  EditorApp.applyImportedCells = applyImportedCells;
  EditorApp.clearGrid = clearGrid;
}
