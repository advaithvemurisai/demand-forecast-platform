// Shapes of the JSON built by scripts/build_web_data.py. Keep in step with it; data.test.ts checks the real files.

export interface Manifest {
  schema: number
  served_method: string
  origins: Record<string, string>
  n_series: number
  products_by_category: Record<string, number>
  twin: boolean
  files: string[]
  bytes: number
}

export interface Summary {
  served_method: string
  selected_bottom_model: string
  n_series: number
  origins: Record<string, string>
  wrmsse_holdout: Record<string, number>
  allocation_vs_pro_rata_ci?: Record<string, { mean: number; lower: number; upper: number; wins: number; n: number }>
  allocation_fairness?: Record<string, string[]>
  twin?: {
    validation_fold: string
    fill_rate: { predicted: number; realised: number; lower: number; upper: number }
    share_realised_in_band: number
    fill_rate_gap: number
    recommended_service: Record<string, number>
    stress_added_lost_sales: Record<string, number>
  }
}

export interface ReconRow { fold: string; method: string; level: string; wmape: number; wrmsse: number; bias: number }
export interface ModelRow { fold: string; model: string; approach: string; level: string; wmape: number; wrmsse: number; bias: number }
export interface CoverageRow { fold: string; level: string; nominal: number; coverage: number; mean_width: number }
export interface SegmentCoverageRow { fold: string; level: string; segment: string; nominal: number; coverage: number; n: number }

export interface AllocationRow {
  node_id: string; dept_id: string; store_id: string; forecast: number; unit_value: number; margin_value?: number
  allocated_quantity: number; pro_rata_quantity: number; expected_fill_rate: number; stockout_risk: number
  supply: number; week_start: string; safety_stock: number; order_up_to: number
}
export interface AllocationBacktestRow {
  fold: string; week: number; policy: string; supply: number; units_fulfilled: number; units_demanded: number
  fill_rate: number; revenue_fulfilled: number; stockout_nodes: number
  units_lost?: number; lost_revenue?: number; min_node_fill?: number
}
export interface NodeFillRow { fold: string; week: number; policy: string; node_id: string; fill_rate: number }
export interface DriftRow {
  node_id: string; demand_psi: number; residual_psi: number; reference_mean_daily: number; current_mean_daily: number
  psi: number; drift: boolean; retrain: boolean; retrain_reasons: string
}
export interface SafetyRow {
  item_id: string; store_id: string; dept_id?: string; cat_id?: string; forecast: number; safety_stock: number; order_up_to: number
  stockout_risk: number; expected_fill_rate?: number; service_level?: number; abc?: string; xyz?: string
}

export interface ColumnarForecast {
  ids: string[]; dates: string[]
  forecast: number[][]; lower_80?: number[][]; upper_80?: number[][]; lower_95: number[][]; upper_95: number[][]
}
export interface BacktestPoint { fold: string; method: string; series_id: string; date: string; forecast: number; actual: number }
export interface LevelForecast { nodes: string[]; production: ColumnarForecast; backtest: BacktestPoint[] }
export interface ForecastIndex { levels: Record<string, string[]>; item_files: { store: string; dept: string; file: string }[] }

export interface DecisionAccuracyRow { fold: string; window_days: number; cat_id: string; velocity: string; wmape: number; bias: number; units: number }
export interface WeekdayBiasRow { fold: string; weekday: string; bias: number }
export interface EventAccuracyRow { fold: string; day_type: string; days: number; item_wmape: number; item_bias: number; total_wmape: number; total_bias: number }
export interface ExceptionRow {
  series_id: string; item_id: string; store_id: string; dept_id: string
  tracking_signal: number; bias_units_per_week: number; direction: string; weekly_impact: number
}
export interface FvaRow { fold: string; scope: string; base_wmape: number; adjusted_wmape: number; fva: number; n: number }
export interface CensoringRow { store_id: string; cat_id: string; flagged_days: number; observed_days: number; censored_share: number }

export interface TwinValidationRow {
  fold: string; store_id: string; policy: string; metric: string; predicted: number; lower: number; upper: number
  realised: number; in_band: boolean; divergence: number
}
export interface TwinTimelineRow { date: string; store_id: string; dept_id: string; policy: string; on_hand: number; demand: number; lost: number }
export interface TwinFrontierRow {
  category: string; service_level: number | null; label: string; fill_rate: number; in_stock_pct: number; inventory_value: number
  lost_sales_value: number; lost_margin: number; holding_cost: number; total_cost: number; total_cost_lower: number; total_cost_upper: number; recommended: boolean
}
export interface TwinStressRow {
  scenario: string; label: string; policy: string; metric: string; baseline: number; baseline_lower: number; baseline_upper: number
  scenario_value: number; scenario_lower: number; scenario_upper: number; delta: number
}
export interface TwinExceptionRow {
  series_id: string; item_id: string; dept_id: string; cat_id: string; store_id: string
  expected_lost_units: number; stockout_probability: number; expected_lost_value: number; policy: string; window_days: number
}

export interface KpiBand { mean: number; lower: number; upper: number }
export type KpiSet = Record<string, KpiBand>
export interface TwinTimelineSet { dept: string[]; on_hand: number[][]; demand: number[][]; lost: number[][] }
export interface TwinOutcome { kpis: KpiSet; timeline: TwinTimelineSet }
export interface TwinResult { policy: string; service: string | number; reps: number; baseline: TwinOutcome; scenario: TwinOutcome | null }
export interface TwinPresets { dates: string[]; results: Record<string, TwinResult> }
export interface TwinPresetIndex {
  stores: string[]
  presets: Record<string, { label: string; detail: string; scenario: Record<string, unknown> }>
  service_grid: number[]
  policies: string[]
}
export interface TwinScenario {
  demand_scale?: number; category?: string | null; days?: [number, number] | null
  dc_factor?: number; dc_days?: [number, number] | null; delay?: number
}
export interface TwinRequest {
  policy: string; service: 'current' | number; rationing: string; scenario: TwinScenario | null; reps: number; seed: number
}
