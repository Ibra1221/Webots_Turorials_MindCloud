import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/mohanad/webots_tutotrial_1/install/webots_pkg_sim'
