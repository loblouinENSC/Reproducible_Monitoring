#!/usr/bin/perl -w

my $usage = 'usage: allen [options] contexts
where options may be:
 -b yyyy-mm-dd: beginning date of the log
 -e yyyy-mm-dd: end date of the log
';

use Time::Local;
#use POSIX qw(strftime);
use Getopt::Std;
our %opts = ();

getopts('b:e:', \%opts) or die $usage;

die "invalid date for option -b"
  if defined($opts{'b'}) && $opts{'b'} !~ /^\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d$/;
die "invalid date for option -e"
  if defined($opts{'e'}) && $opts{'e'} !~ /^\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d$/;

my ($t0, $t1);
my %days = ();
while(<>) {
  chomp;
  my ($ts, $rule, $val, $delta) = split /;/;
  if ($val eq "1") {
    $t0 = &timestamp($ts);
    $t1 = undef;
  } elsif ($val eq "0") {
    $t1 = &timestamp($ts);
    die "undefined t0" if !defined($t0);
    #next if $t0 == 0;
    if ($t0 == 0) { # skip or shorten states at the origin of time (1/1/1970)
      if (defined($opts{'b'})) { $t0 = &timestamp($opts{'b'}); }
      else { next; }
    }
    &count_days($t0, $t1);
    $t0 = undef;
  } # else val eq "?" => ignore
  #print strftime("%j\n", $second, $minute, $hour, $day, $month - 1, $year);
}

# check for an unfinished state at the end
if (defined($opts{'e'}) && defined($t0)) {
  my $t1 = &timestamp($opts{'e'});
  &count_days($t0, $t1)
}

# print result table
for my $k (sort keys %days) {
  print "$k;$days{$k}\n";
}
exit;

sub timestamp() {
  my ($ts) = @_;
  my ($year, $month, $day, $hour, $minute, $second) =
    ($ts =~ /^(\d\d\d\d)-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)/);
  return timelocal($second, $minute, $hour, $day, $month - 1, $year);
}

sub count_days() {
  my ($t0, $t1) = @_;
  my $oneday = 24 * 60 * 60;
  for (my $t = $t0 + $oneday; $t < $t1; $t += $oneday) {
    my ($sec, $min, $hr, $mday, $mon, $yr, $wday, $yday, $isdst) =
      localtime($t);
    $yr += 1900;
    $days{sprintf("%i-%02i", $yr, $mon + 1)}++;
  }
}
