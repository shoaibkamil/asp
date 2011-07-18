# Based loosely on calc example from ply-3.4 distribution

from collections import defaultdict

keywords = ('check',)

tokens = keywords + ('ID','embedded_python')

literals = ['=', '|', '*', '(', ')', ',', '.']

states = (
   ('check','exclusive'),
)

# Borrowed from basiclex.py in ply-3.4 examples
def t_ID(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in keywords:
        t.type = t.value
    if t.value == 'check':
        t.lexer.begin('check')
    return t

def t_check_embedded_python(t):
    r'[^\n]*\n'
    t.lexer.begin('INITIAL')
    return t

t_ignore = " \t"
t_check_ignore = ""

def t_COMMENT(t):
    r'\#[^\n]*\n'
    pass
    
def t_newlines(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")
    
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)

def t_check_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)
    
# Build the lexer
import asp.ply.lex as lex
lex.lex()

# Parsing rules

precedence = (
    ('left','|'),
    ('left','*'),
    ('left','='),
    )

def p_tree_grammar(p):
    '''tree_grammar : rule
                    | tree_grammar rule'''
    if len(p) == 3:
        p[1].append(p[2])
        p[0] = p[1]
    else:
        p[0] = [p[1]]

def p_rule(p):
    '''rule : field_rule
            | alternatives_rule'''
    p[0] = p[1]

def p_field_rule(p):
    'field_rule : ID "(" fields_list ")" checks_list'
    p[0] = FieldRule(p[1], p[3], p[5])

def p_fields_list(p):
    '''fields_list : field
                   | fields_list "," field'''
    if len(p) == 4:
        p[1].append(p[3])
        p[0] = p[1]
    else:
        p[0] = [p[1]]

def p_field(p):
    '''field : ID
             | ID "=" expression'''
    if len(p) == 2:
        p[0] = (p[1],)
    else:
        p[0] = (p[1], p[3])

def p_checks_list(p):
    '''checks_list : 
                   | checks_list check embedded_python'''
    if len(p) == 4:
        p[1].append(p[3])
        p[0] = p[1]
    else:
        p[0] = []

def p_expression(p):
    '''expression : class_name
                  | expression '*'
                  | '(' expression ')'
                  | expression '|' expression '''
    if len(p) == 2:
        p[0] = p[1]
    elif p[2] == '*':
        p[0] = ListOf(p[1])
    elif p[1] == '(':
        p[0] = p[2]
    elif p[2] == '|':
        if isinstance(p[1], OneOf):
            expr_list = p[1].expr_list
            expr_list.append(p[3])
            p[0] = OneOf(expr_list)
        else:
            p[0] = OneOf([p[1], p[3]])

def p_class_name(p):
    '''class_name : ID
                  | class_name '.' ID'''
    if len(p) == 2:
        p[0] = p[1]
    else:
        p[0] = "%s.%s" % (p[1], p[3])

def p_alternatives_rule(p):
    'alternatives_rule : ID "=" alternatives_list'
    p[0] = AlternativesRule(p[1], p[3])

def p_alternatives_list(p):
    '''alternatives_list : class_name
                         | alternatives_list '|' class_name'''
    if len(p) == 4:
        p[1].append(p[3])
        p[0] = p[1]
    else:
        p[0] = [p[1]]

def p_error(p):
    if p:
        print("Syntax error at '%s'" % p.value)
    else:
        print("Syntax error at EOF")


class FieldRule:
    def __init__(self, name, fields_list, checks_list):
        self.name = name
        self.count = 0
        self.fields_list = fields_list
        self.checks_list = checks_list

    def __repr__(self):
        return "%s(%s)" % (self.name, str.join(',', map(str, self.fields_list)))

    def generate(self, parent_map, all_classes):
        field_names = map(lambda x: x[0], self.fields_list)
        return('''
class %s(%s):
    def __init__(self, %s):
        self._fields = (%s,)
        super(%s, self).__init__()
%s
        self.check()

    def check(self):
%s
%s

    def __deepcopy__(self, memo):
        return %s(%s)
        '''
        %
        (self.name, parent_map[self.name],
         str.join(',', field_names),
         str.join(',', map(lambda x: "'%s'" % x, field_names)),
         self.name,
         str.join('\n', map(lambda x: "        self.%s = %s" % (x, x), field_names)),
         str.join('\n', map(lambda x: "        %s" % self.generate_check(x), self.fields_list)),
         str.join('\n', map(lambda x: "        %s" % x.strip(), self.checks_list)),
         self.name,
         str.join(', ', map(lambda x: "copy.deepcopy(self.%s, memo)" % x, field_names))
        )
        )

    def generate_check(self, field):
        return "assert %s, 'Invalid type %%s for field \\'%s\\' of rule \\'%s\\' (value=%%s)' %% (type(self.%s), self.%s)" % (self.generate_check_helper("self.%s" % field[0], field[1] if len(field) > 1 else 'object'), field[0], self.name, field[0], field[0])

    def generate_check_helper(self, name, field_type):
        if isinstance(field_type, OneOf):
            return str.join(' or ', map(lambda x: self.generate_check_helper(name, x), field_type.expr_list))
        elif isinstance(field_type, ListOf):
            var_name = self.fresh_identifier()
            return "len(filter(lambda %s: not (%s), %s)) == 0" % (var_name, self.generate_check_helper(var_name, field_type.expr), name)
        else:
            return "isinstance(%s, %s)" % (name, field_type)

    def get_classes(self):
        result_list = [self.name]
        for x in self.fields_list:
            if len(x) > 1:
                self.get_classes_helper(result_list, x[1])
        return result_list

    def get_classes_helper(self, result_list, field_type):
        if isinstance(field_type, OneOf):
            for x in field_type.expr_list:
                self.get_classes_helper(result_list, x)
        elif isinstance(field_type, ListOf):
            self.get_classes_helper(result_list, field_type.expr)
        else:
            result_list.append(field_type)

    def get_parent_map(self):
        return dict()

    def fresh_identifier(self):
        self.count += 1
        return 'x%d' % self.count

class AlternativesRule:
    def __init__(self, name, alternatives):
        self.name = name
        self.alternatives = alternatives

    def __repr__(self):
        return "%s = %s" % (self.name, str.join(' | ', map(str, self.alternatives)))

    def generate(self, parent_map, all_classes):
        return '''
class %s(%s):
    def __init__(self):
        super(%s, self).__init__()
''' % (self.name, parent_map[self.name], self.name)

    def get_classes(self):
        result = [self.name]
        result.extend(self.alternatives)
        return result

    def get_parent_map(self):
        return dict(map(lambda x: (x, self.name), self.alternatives))

class ListOf:
    def __init__(self, expr):
        self.expr = expr

    def __repr__(self):
        return "ListOf(%s)" % (self.expr)

class OneOf:
    def __init__(self, expr_list):
        self.expr_list = expr_list

    def __repr__(self):
        return "OneOf(%s)" % (self.expr_list)

def generate_checker_class(checker, rules):
    result = "class %s(ast.NodeVisitor):" % checker
    for rule in rules:
        result += '''
    def visit_%s(self, node):
        node.check()
        self.generic_visit(node)
        ''' % rule
    return result

def parse(tree_grammar, global_dict, checker=None):
    import ply.yacc as yacc
    yacc.yacc()
    result = yacc.parse(tree_grammar)

    parent_map = defaultdict(lambda: 'ast.AST')
    for rule in result:
        rule_map = rule.get_parent_map()
        assert len(set(parent_map.keys()) & set(rule_map.keys())) == 0, 'Same class occured in two alternative rules, but can only have one base class'
        parent_map.update(rule_map)

    program = "import copy\n"

    classes_with_rules = []
    all_classes = []
    for rule in result:
        classes_with_rules.append(rule.name)
        all_classes.extend(rule.get_classes())
    all_classes = set(filter(lambda x: not("." in x), all_classes))
    classes_with_rules = set(classes_with_rules)

    for rule in result:
        program += rule.generate(parent_map, all_classes)

    for x in all_classes - classes_with_rules:
        program += '''
class %s(%s):
    def __init__(self):
        super(%s, self).__init__()
''' % (x, parent_map[x], x)

    if checker != None:
        program = "import ast\n" + program + "\n" + generate_checker_class(checker, classes_with_rules) + "\n"

    exec(program, global_dict)
