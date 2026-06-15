import { useQuery } from "@tanstack/react-query";

export function usePolling<T>(
  key: unknown[],
  fn: () => Promise<T>,
  isActive: (data: T | undefined) => boolean,
  intervalMs = 2000
) {
  return useQuery({
    queryKey: key,
    queryFn: fn,
    refetchInterval: (query) => (isActive(query.state.data) ? intervalMs : false),
  });
}
