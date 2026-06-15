export default function Spinner({ size = 6 }: { size?: number }) {
  return (
    <div
      className={`w-${size} h-${size} rounded-full border-2 border-slate-200 border-t-blue-600 animate-spin`}
    />
  );
}
