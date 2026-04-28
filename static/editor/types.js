// @ts-check

/**
 * Shared JSDoc typedefs for the editor scripts.
 * This file has no runtime exports and exists only to centralize types.
 */

/**
 * Paint mode chosen in the toolbar.
 * @typedef {'cycle' | 'transparent' | 'white' | 'black'} PaintMode
 */

/**
 * Active drawing tool in the editor.
 * @typedef {'brush' | 'fill' | 'line'} ActiveTool
 */

/**
 * Logical grid coordinate plus flattened array index.
 * @typedef {{ col: number, row: number, idx: number }} Cell
 */

/**
 * Grid totals used by the status readout.
 * @typedef {{ transparent: number, white: number, black: number }} CellCounts
 */

/**
 * Mutable state shared by the editor controller and renderer.
 * @typedef {{
 *   cells: Uint8Array,
 *   underlayImage: HTMLImageElement | null,
 *   underlayName: string,
 *   underlayStatusMessage: string,
 *   importedGridName: string,
 *   gridImportStatusMessage: string,
 *   paintMode: PaintMode,
 *   activeTool: ActiveTool,
 *   isPainting: boolean,
 *   isDrawingLine: boolean,
 *   lineStartCell: Cell | null,
 *   linePreviewCell: Cell | null,
 *   linePreviewShift: boolean,
 *   gestureChanged: boolean,
 *   historyBeforeGesture: Uint8Array | null,
 *   lastPaintedIndex: number,
 *   undoStack: Uint8Array[],
 *   redoStack: Uint8Array[],
 *   zoomScale: number,
 * }} EditorState
 */

/**
 * View-only values derived from the current control inputs.
 * @typedef {{
 *   underlayOpacity: number,
 *   gridFillOpacity: number,
 *   gridOpacity: number,
 * }} ViewState
 */

/**
 * Export settings captured at download time.
 * @typedef {{
 *   scale: number,
 *   includeUnderlay: string,
 *   underlayOpacity: number,
 * }} ExportOptions
 */

/**
 * Decoded payload returned from grid image import.
 * @typedef {{
 *   cells: Uint8Array,
 *   width: number,
 *   height: number,
 *   scale: number,
 * }} ImportedGrid
 */

/**
 * DOM references required by the editor runtime.
 * @typedef {{
 *   canvas: HTMLCanvasElement,
 *   canvasScroll: HTMLElement,
 *   canvasShell: HTMLElement,
 *   zoomOutBtn: HTMLButtonElement,
 *   zoomFitBtn: HTMLButtonElement,
 *   zoomInBtn: HTMLButtonElement,
 *   zoomValue: HTMLElement,
 *   imageInput: HTMLInputElement,
 *   gridImageInput: HTMLInputElement,
 *   removeUnderlayBtn: HTMLButtonElement,
 *   underlayOpacity: HTMLInputElement,
 *   opacityValue: HTMLElement,
 *   gridFillOpacity: HTMLInputElement,
 *   gridFillOpacityValue: HTMLElement,
 *   gridOpacity: HTMLInputElement,
 *   gridOpacityValue: HTMLElement,
 *   exportScale: HTMLSelectElement,
 *   includeUnderlay: HTMLSelectElement,
 *   clearBtn: HTMLButtonElement,
 *   exportBtn: HTMLButtonElement,
 *   undoBtn: HTMLButtonElement,
 *   redoBtn: HTMLButtonElement,
 *   status: HTMLElement,
 *   underlayStatus: HTMLElement,
 *   gridImportStatus: HTMLElement,
 *   cursorReadout: HTMLElement,
 *   swatches: HTMLButtonElement[],
 *   toolButtons: HTMLButtonElement[],
 * }} EditorElements
 */

/**
 * Rendering API exposed to the controller.
 * @typedef {{
 *   refresh: (state: EditorState, view: ViewState, cursorText: string) => void,
 *   renderExportCanvas: (state: EditorState, options: ExportOptions) => HTMLCanvasElement,
 *   syncCanvasLayout: (zoomScale: number) => void,
 *   updateCursor: (text: string) => void,
 * }} EditorRenderer
 */

/**
 * Public handle returned by editor bootstrap.
 * @typedef {{ state: EditorState }} EditorController
 */

/**
 * Shared global namespace used by the classic editor scripts.
 * @typedef {{
 *   COLS: number,
 *   ROWS: number,
 *   PREVIEW_CELL: number,
 *   PREVIEW_W: number,
 *   PREVIEW_H: number,
 *   MAX_HISTORY: number,
 *   MIN_ZOOM: number,
 *   MAX_ZOOM: number,
 *   ZOOM_STEP: number,
 *   VIEWPORT_PADDING: number,
 *   STATE_TRANSPARENT: number,
 *   STATE_WHITE: number,
 *   STATE_BLACK: number,
 *   createEditorState: () => EditorState,
 *   indexOf: (col: number, row: number) => number,
 *   clamp: (value: number, min: number, max: number) => number,
 *   setPaintMode: (state: EditorState, paintMode: PaintMode) => void,
 *   setActiveTool: (state: EditorState, activeTool: ActiveTool) => void,
 *   countCells: (state: EditorState) => CellCounts,
 *   setZoom: (state: EditorState, nextZoom: number) => void,
 *   zoomIn: (state: EditorState) => void,
 *   zoomOut: (state: EditorState) => void,
 *   fitZoom: (state: EditorState) => void,
 *   undo: (state: EditorState) => boolean,
 *   redo: (state: EditorState) => boolean,
 *   setUnderlay: (state: EditorState, image: HTMLImageElement, name: string) => void,
 *   clearUnderlay: (state: EditorState) => void,
 *   setUnderlayStatusMessage: (state: EditorState, message: string) => void,
 *   getUnderlayStatus: (state: EditorState) => string,
 *   clearImportedGridLabel: (state: EditorState) => void,
 *   setImportedGridLabel: (state: EditorState, label: string) => void,
 *   setGridImportStatusMessage: (state: EditorState, message: string) => void,
 *   getGridImportStatus: (state: EditorState) => string,
 *   nextState: (current: number) => number,
 *   getPaintTarget: (state: EditorState, current: number, shiftKey?: boolean) => number,
 *   beginBrushStroke: (state: EditorState) => void,
 *   applyBrushAtCell: (state: EditorState, cell: Cell | null, shiftKey: boolean) => boolean,
 *   endBrushStroke: (state: EditorState) => boolean,
 *   applyFillAtCell: (state: EditorState, cell: Cell | null, shiftKey: boolean) => boolean,
 *   getLineCells: (start: Cell, end: Cell) => Cell[],
 *   startLine: (state: EditorState, cell: Cell | null, shiftKey: boolean) => boolean,
 *   updateLinePreview: (state: EditorState, cell: Cell | null, shiftKey: boolean) => boolean,
 *   finishLine: (state: EditorState, endCell: Cell | null, shiftKey: boolean) => boolean,
 *   cancelLine: (state: EditorState) => void,
 *   applyImportedCells: (state: EditorState, cells: Uint8Array, label: string) => boolean,
 *   clearGrid: (state: EditorState) => boolean,
 *   createEditorRenderer: (elements: EditorElements) => EditorRenderer,
 *   loadImageViaFileReader: (file: File) => Promise<HTMLImageElement>,
 *   importGridCellsFromImage: (img: HTMLImageElement, sourceLabel: string) => ImportedGrid,
 *   downloadCanvasAsPng: (canvasToDownload: HTMLCanvasElement) => void,
 *   createEditorController: (root?: Document) => EditorController,
 * }} EditorAppNamespace
 */

/**
 * Window shape used by the classic editor scripts.
 * @typedef {Window & typeof globalThis & { EditorApp: EditorAppNamespace }} EditorWindow
 */
