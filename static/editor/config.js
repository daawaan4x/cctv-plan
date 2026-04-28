/// <reference path="./types.js" />
// @ts-check

{
  /** @type {EditorWindow} */
  const editorWindow = /** @type {EditorWindow} */ (window);
  editorWindow.EditorApp = /** @type {EditorAppNamespace} */ (editorWindow.EditorApp || {});
  const EditorApp = editorWindow.EditorApp;

  /**
   * Grid dimensions in editable cells.
   * @type {number}
   */
  const COLS = 210;

  /**
   * Grid dimensions in editable cells.
   * @type {number}
   */
  const ROWS = 90;

  /**
   * Preview scale for one logical grid cell on the editor canvas.
   * @type {number}
   */
  const PREVIEW_CELL = 8;

  /**
   * Backing canvas width in device-independent pixels.
   * @type {number}
   */
  const PREVIEW_W = COLS * PREVIEW_CELL;

  /**
   * Backing canvas height in device-independent pixels.
   * @type {number}
   */
  const PREVIEW_H = ROWS * PREVIEW_CELL;

  /**
   * Maximum number of undo snapshots retained in memory.
   * @type {number}
   */
  const MAX_HISTORY = 120;

  /**
   * The editor never zooms below the baseline fitted size.
   * @type {number}
   */
  const MIN_ZOOM = 1;

  /**
   * Upper zoom bound for the editor viewport.
   * @type {number}
   */
  const MAX_ZOOM = 4;

  /**
   * Multiplicative zoom step used by the zoom controls.
   * @type {number}
   */
  const ZOOM_STEP = 1.25;

  /**
   * Padding reserved inside the scroll viewport when fitting the canvas.
   * @type {number}
   */
  const VIEWPORT_PADDING = 32;

  /**
   * Cell state values stored in the compact `Uint8Array` grid.
   * @type {number}
   */
  const STATE_TRANSPARENT = 0;

  /**
   * Cell state values stored in the compact `Uint8Array` grid.
   * @type {number}
   */
  const STATE_WHITE = 1;

  /**
   * Cell state values stored in the compact `Uint8Array` grid.
   * @type {number}
   */
  const STATE_BLACK = 2;

  EditorApp.COLS = COLS;
  EditorApp.ROWS = ROWS;
  EditorApp.PREVIEW_CELL = PREVIEW_CELL;
  EditorApp.PREVIEW_W = PREVIEW_W;
  EditorApp.PREVIEW_H = PREVIEW_H;
  EditorApp.MAX_HISTORY = MAX_HISTORY;
  EditorApp.MIN_ZOOM = MIN_ZOOM;
  EditorApp.MAX_ZOOM = MAX_ZOOM;
  EditorApp.ZOOM_STEP = ZOOM_STEP;
  EditorApp.VIEWPORT_PADDING = VIEWPORT_PADDING;
  EditorApp.STATE_TRANSPARENT = STATE_TRANSPARENT;
  EditorApp.STATE_WHITE = STATE_WHITE;
  EditorApp.STATE_BLACK = STATE_BLACK;
}
