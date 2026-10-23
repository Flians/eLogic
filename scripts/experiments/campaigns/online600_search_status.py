"""Conservative read-only search-completion view; never changes a committed mapping result."""
def classify(row):
    parent=row.get('mapping',{});native=row.get('native',{})
    outer_timeout=bool(parent.get('timeout'))
    native_timeout=bool(native.get('search_deadline_hit'))
    code=native.get('native_exit_code')
    finished=parent.get('exit_code')==0 and code==0 and not(outer_timeout or native_timeout)
    return dict(search_finished=finished,search_incomplete=not finished,
                outer_timeout=outer_timeout,native_deadline_flag=native_timeout,native_exit_code=code,
                native_nonzero_exit=code is not None and code!=0,
                frontend_exit_code=parent.get('exit_code'),
                optimized_output_independent_CEC=row.get('status')=='ok' and row.get('measured',{}).get('cec')==1,
                original_reports_unchanged=True)
