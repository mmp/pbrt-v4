"""
LLDB Formatters

1. Load Manually (every time)
    use lldb command 'command script import [PATH]/pbrt_lldbdataformatters.py' (without the quotes)

2. Load Automatically (add to ~/.lldbinit)
    1) create ~/.lldbinit file if there isn't
    2) open ~/.lldbinit
    3) add 'command script import [PATH]/pbrt_lldbdataformatters.py' to the file (without the quotes)
    4) restart xcode
"""

import lldb;
import lldb.formatters.Logger;

# Uncomment this to debug
# lldb.formatters.Logger._lldb_formatters_debug_level=1

# TaggedPointer
#   seems GetTemplateArgumentType() does work with Variadic Templates, have to parse them into list
class TaggedPointerSynthProvider:
    def __init__(self, valobj, dirt):
        logger = lldb.formatters.Logger.Logger()
        self.valobj = valobj
        typeName=self.valobj.GetTypeName()
        leftBracket = typeName.find('<')
        rightBracket = typeName.find('>')
        packedTypes = typeName[leftBracket+1:rightBracket]
        logger >> "PackedType: " + str(packedTypes)
        self.type_strings = packedTypes.split(',')
        logger >> "PackedTypeList: " + str(self.type_strings)
        self.update()

    def has_children(self):
        return True

    def num_children(self):
        return 2

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        logger = lldb.formatters.Logger.Logger()
        if index == 0:
            return self.valobj.CreateValueFromExpression('tag', str(self.tag))
        if index == 1:
            if self.tag > 0:
                expr = '(' + self.type + '*)' + str(self.ptr)
                logger >> 'expr ' + expr
                return self.valobj.CreateValueFromExpression('ptr', expr)
            else:
                return self.valobj.CreateValueFromExpression('ptr', 'null')
        return None

    def update(self):
        bits = self.valobj.GetChildMemberWithName('bits').GetValueAsUnsigned()
        self.tag = (bits & 0xFE00000000000000) >> 57
        if self.tag > 0 and self.tag <= len(self.type_strings):
            self.type = self.type_strings[self.tag-1]
        else:
            self.type = 'void'
        self.ptr = (bits& 0x01FFFFFFFFFFFFFF)

# Optional
class OptionalSynthProvider:
    def __init__(self, valobj, dirt):
        self.valobj = valobj
        self.update()

    def has_children(self):
        return self.set.GetValueAsUnsigned() != 0

    def num_children(self):
        if self.set.GetValueAsUnsigned() == 0:
            return 0
        else:
            return 1

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if index == 0:
            expr = '*(' + self.type.GetName() + '*)' + str(self.ptr.GetValueAsUnsigned())
            return self.valobj.CreateValueFromExpression('ptr', str(expr))
        return None

    def update(self):
        try:
            self.set = self.valobj.GetChildMemberWithName('set')
            self.type = self.extract_type()
            self.ptr = self.valobj.GetChildMemberWithName('optionalValue').AddressOf()
        except:
            pass

    def extract_type(self):
        arrayType = self.valobj.GetType().GetUnqualifiedType()
        if arrayType.IsReferenceType():
            arrayType = arrayType.GetDereferencedType()
        elif arrayType.IsPointerType():
            arrayType = arrayType.GetPointeeType()
        if arrayType.GetNumberOfTemplateArguments() > 0:
            elementType = arrayType.GetTemplateArgumentType(0)
        else:
            elementType = None
        return elementType

def OptionalSummaryProvider(valobj, internal_dict):
    set = valobj.GetNumChildren()
    if set:
        return 'set'
    return 'null'


# Span
class SpanSynthProvider:
    def __init__(self, valobj, dirt):
        self.valobj = valobj
        self.update()

    def has_children(self):
        return True

    def num_children(self):
        return self.size

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        offset = index * self.type_size
        return self.ptr.CreateChildAtOffset('['+str(index)+']', offset, self.date_type)

    def update(self):
        self.size = self.valobj.GetChildMemberWithName('n').GetValueAsUnsigned()
        self.ptr = self.valobj.GetChildMemberWithName('ptr')
        valType = self.valobj.GetType()
        if valType.IsReferenceType():
            valType = valType.GetDereferencedType()
        self.date_type = valType.GetTemplateArgumentType(0)
        self.type_size = self.date_type.GetByteSize()

def SpanSummaryProvider(valobj, internal_dict):
    length = valobj.GetNumChildren()
    return 'size = %d' % length


# Array
class ArraySynthProvider:
    def __init__(self, valobj, dirt):
        self.valobj = valobj
        self.update()

    def has_children(self):
        return True

    def num_children(self):
        return self.size

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        if self.data_type == None:
            return None
        offset = index * self.type_size
        return self.values.CreateChildAtOffset('['+str(index)+']', offset, self.data_type)

    def update(self):
        try:
            self.values = self.valobj.GetChildMemberWithName('values')
            valType = self.values.GetType()
            if valType.IsReferenceType():
                valType = valType.GetDereferencedType()
            if valType.IsArrayType():
                arraySize = valType.GetByteSize()
                self.data_type = valType.GetArrayElementType()
                self.type_size = self.data_type.GetByteSize()
                self.size = arraySize/self.type_size
            else:
                self.data_type = None
                self.type_size = 0
                self.size = 0
        except:
            pass
  
def ArraySummaryProvider(valobj, internal_dict):
    length = valobj.GetNumChildren()
    return 'size = %d' % length

# Vector
class VectorSynthProvider:
    def __init__(self, valobj, dirt):
        self.valobj = valobj
        self.update()

    def has_children(self):
        return True

    def num_children(self):
        return self.size.GetValueAsUnsigned()

    def get_child_index(self, name):
        try:
            return int(name.lstrip('[').rstrip(']'))
        except:
            return -1

    def get_child_at_index(self, index):
        if index < 0:
            return None
        if index >= self.num_children():
            return None
        if self.data_type == None:
            return None
        offset = index * self.type_size
        return self.ptr.CreateChildAtOffset('['+str(index)+']', offset, self.data_type)

    def update(self):
        try:
            self.size = self.valobj.GetChildMemberWithName('nStored')
            self.capacity = self.valobj.GetChildMemberWithName('nAlloc')
            self.ptr = self.valobj.GetChildMemberWithName('ptr')
            self.data_type = self.ptr.GetType()
            if self.data_type.IsPointerType():
                self.data_type = self.data_type.GetPointeeType()
            self.type_size = self.data_type.GetByteSize()
        except:
            pass

def VectorSummaryProvider(valobj, internal_dict):
    length = valobj.GetNumChildren()
    return 'size = %d' % length


def __lldb_init_module(debugger, internal_dict):
    debugger.HandleCommand('type synthetic add -l pbrt_lldbdataformatters.TaggedPointerSynthProvider -x "^pbrt::TaggedPointer<.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type synthetic add -l pbrt_lldbdataformatters.OptionalSynthProvider -x "^pstd::optional<.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type summary add -F pbrt_lldbdataformatters.OptionalSummaryProvider -e -x "^pstd::optional<.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type synthetic add -l pbrt_lldbdataformatters.SpanSynthProvider -x "^pstd::span<.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type summary add -F pbrt_lldbdataformatters.SpanSummaryProvider -e -x "^pstd::span<.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type synthetic add -l pbrt_lldbdataformatters.ArraySynthProvider -x "^pstd::array<.+,.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type summary add -F pbrt_lldbdataformatters.ArraySummaryProvider -e -x "^pstd::array<.+,.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type synthetic add -l pbrt_lldbdataformatters.VectorSynthProvider -x "^pstd::vector<.+,.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand('type summary add -F pbrt_lldbdataformatters.VectorSummaryProvider -e -x "^pstd::vector<.+,.+>$" -w pbrt_lldbdataformatters')
    debugger.HandleCommand("type category enable pbrt_lldbdataformatters")
