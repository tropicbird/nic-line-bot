import sys
import pkg_resources

print(f'現在のpythonのバージョン：{sys.version}')

print(f'現在のpythonの場所：{sys.executable}')

installed_packages = pkg_resources.working_set
installed_packages_list = sorted(["%s==%s" % (i.key, i.version)
   for i in installed_packages])
print(f'現在のpythonにインストールされているパッケージ一覧：{installed_packages_list}')