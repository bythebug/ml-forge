type Variant = "green" | "yellow" | "red" | "blue" | "gray";

const styles: Record<Variant, string> = {
  green: "bg-green-50 text-green-700 border-green-200",
  yellow: "bg-yellow-50 text-yellow-700 border-yellow-200",
  red: "bg-red-50 text-red-700 border-red-200",
  blue: "bg-blue-50 text-blue-700 border-blue-200",
  gray: "bg-slate-50 text-slate-600 border-slate-200",
};

interface BadgeProps {
  label: string;
  variant?: Variant;
  pulse?: boolean;
}

export default function Badge({ label, variant = "gray", pulse = false }: BadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-xs font-medium ${styles[variant]}`}
    >
      {pulse && (
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-yellow-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-yellow-500" />
        </span>
      )}
      {label}
    </span>
  );
}
