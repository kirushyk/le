# command script import le/tools/debug/le_lldb.py
# type summary add -F le_lldb.LeShapeSummary LeShape
# type summary add -F le_lldb.LeTensorSummary LeTensor

import lldb

def LeShapeSummary(value, internal_dict):
    num_dimensions = value.GetChildMemberWithName("num_dimensions").GetValueAsUnsigned()
    summary = ""
    for i in range(0, num_dimensions):
        summary += str(value.GetChildMemberWithName("sizes").GetChildAtIndex(i, lldb.eDynamicCanRunTarget, True).GetValueAsUnsigned())
        if i < (num_dimensions - 1):
            summary += "x"
    summary += ""
    return summary

def LeTensorSummary(value, internal_dict):
    return LeShapeSummary(value.GetChildMemberWithName("shape"), internal_dict)
